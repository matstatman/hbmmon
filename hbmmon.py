#!/usr/bin/env python3

import argparse
import os
import struct
import subprocess
import sys
import time
from collections import deque

import hbmmap as hm

FBPA_N, FBP_N = 24, 12
DENSITY_GEO = {          # code -> (Gb per die, dies per stack)
    0x1: (1, 8), 0x2: (2, 8), 0x3: (4, 8), 0x4: (8, 8), 0x5: (6, 8),
    0x6: (8, 8), 0x8: (12, 8), 0x9: (8, 12), 0xA: (16, 8),
    0xB: (12, 12), 0xC: (16, 12),
}
MFR_NAMES = {0x1: "Samsung", 0x6: "SK Hynix", 0xF: "Micron"}
DENSITY_NAMES = {
    0x1: "1 Gb 4-High", 0x2: "2 Gb 4-High", 0x3: "4 Gb 4-High",
    0x4: "8 Gb 4-High", 0x5: "6 Gb 4-High", 0x6: "8 Gb 4-High (256B)",
    0x8: "12 Gb 4-High", 0x9: "8 Gb 6-High", 0xA: "16 Gb 8-High",
    0xB: "12 Gb 6-High", 0xC: "16 Gb 6-High",
}
ADDR_MODE_NAMES = {0: "single-channel", 1: "unknown channel",
                   2: "16-bit mode"}


def id_fields(w82):
    return {
        "gen2_test": (w82 >> 81) & 1,
        "ecc": (w82 >> 80) & 1,
        "density": (w82 >> 76) & 0xF,
        "mfr": (w82 >> 72) & 0xF,
        "year": 2011 + ((w82 >> 60) & 0xFF),
        "week": (w82 >> 52) & 0xFF,
        "serial": (w82 >> 18) & 0x3FFFFFFFF,   # 34-bit field [51:18]
        "addr_mode": (w82 >> 16) & 3,
        "ch_avail": (w82 >> 8) & 0xFF,
        "stack_h": (w82 >> 7) & 1,
        "model": w82 & 0x7F,
    }


def gib_per_fbpa(fields):
    """GiB per FBPA from a decoded DEVICE_ID, or None if unknown code."""
    geo = DENSITY_GEO.get(fields["density"]) if fields else None
    if not geo:
        return None
    return geo[0] * geo[1] / 32.0


# short box labels (hbmmap.stack_pair's 18-col box: 14 usable)
def short_size(m):
    geo = DENSITY_GEO.get(m["density"]) if m else None
    return f"{geo[0]}Gb" if geo else "?"


def short_hi(m):
    if not m:
        return "?"
    return "8Hi" if m["stack_h"] else "4Hi"


def short_mfr(m):
    return MFR_NAMES.get(m["mfr"], "?") if m else "?"


def short_ecc(m):
    if not m:
        return "?"
    return "ECC" if m["ecc"] else "no ECC"


# ---------------------------------------------------------------------------
# the DEVICE_ID sweep (write-gated) + topology derivation

def capture_once(rd, wr, gp, f):
    """One §11.8 arm-capture of one live FBPA -> decoded 82-bit DEVICE_ID
    or None. Failure modes (all printed, all retryable by re-capturing):
      * DROPPED / PARTIAL   — write didn't land (ISR in the arm window);
      * ISR-RACE            — the Falcon's poller replaced the WIR
                              mid-capture (SHWIR left the echo);
      * no 3-word window    — SHWIR held the echo, but the reads landed
                              in the inter-burst gap (DATA latch holds
                              the frame's idle word, 0x8a000000 class);
      * AMBIGUOUS           — >1 ordering fits the 14-bit law
                              (~1-in-16k per wrong ordering).
    wdr_arm_capture itself re-arms up to 2x internally on ISR-RACE /
    DROPPED/PARTIAL before giving up.
    """
    words, status, note = gp.wdr_arm_capture(
        rd, wr, f, 0x0E, 6, True,
        mode_arm=gp.WDR_MODE_STREAM, channel=gp.WDR_WDR_CH)
    tag = f"  fbpa={f:2d}  {status}"
    if words is None or status != "OK":
        print(f"{tag}  {note}")
        return None
    cands = None
    for k in range(len(words) - 2):
        c = gp._wdr_frame_order(words[k:k + 3])
        if c:
            cands = c
            break
    if not cands:
        print(f"{tag}  no 3-word window (inter-burst reads? data: "
              f"{' '.join(f'0x{w[0]:08x}' for w in words)})")
        return None
    if len(cands) > 1:
        print(f"{tag}  AMBIGUOUS ({len(cands)} orderings fit)")
        return None
    X, Y, Z = cands[0]
    ID = (Z << 50) | (Y << 32) | X
    print(f"{tag}  DEVICE_ID=0x{ID:021x}")
    return ID


def id_sweep(rd, wr, gp, live, max_retakes=2, max_rounds=3, budget=96):
    """§11.8 DEVICE_ID arm-capture over every live FBPA.

    Returns {DEVICE_ID_82: [fbpa, ...]}, or None if the abort criteria
    tripped (sentinel change / new Xid) — same criteria as
    gpc_probe --wdr_deviceid, plus two retry layers on top of the
    per-capture recipe (unchanged): undecoded FBPAs are re-taken up to
    max_retakes times and then re-swept in further rounds up to
    max_rounds, so a stochastic ISR-race pass cannot leave a stack with
    zero decoded FBPAs. budget caps total arm-captures so a
    pathological Falcon schedule can't hammer a live GPU.
    """
    x0 = subprocess.run(["dmesg"], capture_output=True, text=True,
                        timeout=10).stdout.splitlines()
    s0 = gp.hbm_sentinels(rd)
    groups = {}
    undecoded = list(live)
    n_cap = {"n": 0}

    def armed(f):
        """Up to max_retakes captures of one FBPA, under the budget."""
        ID = None
        for _ in range(max_retakes):
            if n_cap["n"] >= budget:
                break
            n_cap["n"] += 1
            ID = capture_once(rd, wr, gp, f)
            if ID is not None:
                break
        return ID

    for rnd in range(1, max_rounds + 1):
        if not undecoded:
            break
        if rnd > 1:
            print(f"  round {rnd}/{max_rounds}: re-sweeping "
                  f"{len(undecoded)} undecoded: {undecoded}")
        for f in list(undecoded):
            if n_cap["n"] >= budget:
                print(f"  budget of {budget} arm-captures reached; "
                      f"{len(undecoded)} undecoded")
                undecoded = []
                break
            ID = armed(f)
            if ID is not None:
                groups.setdefault(ID, []).append(f)
                undecoded.remove(f)
            if gp.hbm_sentinels(rd) != s0:
                print(f"!! ABORT after fbpa={f}: sentinel change — "
                      "stop, reboot the GPU")
                return None
            dx = gp.xid_since(x0)
            if dx:
                print(f"!! ABORT after fbpa={f}: new Xid {dx} — "
                      "stop, reboot the GPU")
                return None
    fin = gp.hbm_sentinels(rd)
    n_dec = len(live) - len(undecoded)
    print(f"  sentinels {'CLEAN' if fin == s0 else 'CHANGED — reboot advised'}"
          f" · decoded {n_dec}/{len(live)} live FBPAs"
          f" · {len(groups)} distinct DEVICE_ID(s) · {n_cap['n']} "
          f"arm-captures")
    if undecoded:
        print(f"  !! still undecoded: {undecoded} (re-run; each pass is a "
              "fresh stochastic sample of the ISR schedule)")
    return groups


def derive_stacks(groups, live_set, fbp_dis, fbp_def):
    """Derive the 12-FBP card map from hardware state — no hardcoded map.

    Live stacks: live FBPAs grouped by identical 82-bit DEVICE_ID (§11.5).
    A live stack's full footprint is the live FBPAs of its FBP (both
    FBPAs of a live FBP tap the same two dies) — that expansion keeps a
    partial decode in the right place; "decoded" keeps the raw sweep
    result for display. Dead stack slots: the dead FBPAs' FBP indices,
    sorted, chunked in twos (pairing not recoverable at runtime; on the
    170HX this yields the measured {1,4}/{6,11}). Letters: live stacks
    by min FBPA (A..), then dead slots by FBP order. Side: left if all
    of the stack's FBP are in the first half of the 12.
    """
    stacks = []
    for ID, fbpas in sorted(groups.items(), key=lambda kv: min(kv[1])):
        fbpas = sorted(fbpas)
        fbps = sorted({f // 2 for f in fbpas})
        full = sorted({fb for fp in fbps for fb in (2 * fp, 2 * fp + 1)}
                      & live_set)
        stacks.append({"ID": ID, "fbpas": full, "decoded": fbpas,
                       "fbps": fbps,
                       "meta": id_fields(ID), "state": "live"})
    dead_fbp = sorted({f // 2 for f in range(FBPA_N) if f not in live_set})
    for i in range(0, len(dead_fbp), 2):
        pair = dead_fbp[i:i + 2]
        stacks.append({
            "ID": None,
            "fbpas": sorted({fb for fp in pair for fb in (2 * fp, 2 * fp + 1)}
                            & live_set),
            "decoded": [], "fbps": pair, "meta": None,
            "state": ("defective"
                      if any((fbp_def >> f) & 1 for f in pair)
                      else "harvested"),
        })
    for i, s in enumerate(stacks):
        s["letter"] = "ABCDEFGH"[i]
        s["side"] = "L" if max(s["fbps"]) < FBP_N // 2 else "R"
    return stacks


def ro_stacks(live_set, fbp_dis, fbp_def):
    """Read-only fallback: 6 boxes in assumed contiguous FBP pairs
    (0,1) (2,3) ... — the pairing is NOT verified without the sweep, so
    boxes carry no stack letter and no identity text."""
    stacks = []
    for i in range(FBP_N // 2):
        f0, f1 = 2 * i, 2 * i + 1
        fbpas = sorted({2 * f0, 2 * f0 + 1, 2 * f1, 2 * f1 + 1} & live_set)
        # CFG1 liveness is the ground truth: a pair with any live FBPA
        # draws live (its fused-off FBP still shows dim inside the box);
        # only fully-dark pairs take the fuse state. On the 170HX every
        # assumed pair mixes one live + one dead FBP, so this matters.
        if fbpas:
            state = "live"
        elif (fbp_def >> f0) & 1 or (fbp_def >> f1) & 1:
            state = "defective"
        elif (fbp_dis >> f0) & 1 or (fbp_dis >> f1) & 1:
            state = "harvested"
        else:
            state = "harvested"       # fuses clean but dark — treat dark
        stacks.append({
            "ID": None,
            "fbpas": fbpas,
            "decoded": [], "fbps": (f0, f1), "meta": None,
            "state": state, "letter": "?",
            "side": "L" if f1 < FBP_N // 2 else "R",
        })
    return stacks


# ---------------------------------------------------------------------------
# per-tick data (reads only; same registers as hbmmap.collect)

_PCI_ID_CACHE = {}


def _pci_id(bdf_s):
    """PCI vendor:device from sysfs (e.g. "10de:20c2") — the kernel's
    identity for the card, read not assumed. Cached: it never changes."""
    if bdf_s not in _PCI_ID_CACHE:
        vid = did = None
        d = f"/sys/bus/pci/devices/{bdf_s.lower()}"
        try:
            with open(f"{d}/vendor") as fh:
                vid = fh.read().strip()
            with open(f"{d}/device") as fh:
                did = fh.read().strip()
        except OSError:
            pass
        if vid and did:
            vid = vid[2:] if vid.startswith("0x") else vid
            did = did[2:] if did.startswith("0x") else did
            _PCI_ID_CACHE[bdf_s] = f"{vid}:{did}"
        else:
            _PCI_ID_CACHE[bdf_s] = None
    return _PCI_ID_CACHE[bdf_s]


def collect(rd, bdf_s, smi):
    """Fuses + TPC status + CFG1 live map + per-FBPA temperatures.
    Reads only — the topology comes from the startup sweep."""
    sm = smi.get(bdf_s.lower(), {})
    gpc_dis = rd(hm.OFF_GPC_DIS)
    gpc_def = rd(hm.OFF_GPC_DEF)
    fbp_dis = rd(hm.OFF_FBP_DIS)
    fbp_def = rd(hm.OFF_FBP_DEF)
    nvl_dis = rd(hm.OFF_NVL_DIS)
    nvl_def = rd(hm.OFF_NVL_DEF)
    tpc = [rd(hm.OFF_TPC_STATUS + 4 * i) & 0xFF for i in range(8)]
    live = set(hm.live_fbpas(rd))
    temps = {}
    for f in sorted(live):
        data = rd(hm.FBPA_UC_BASE + f * hm.FBPA_UC_STRIDE + hm.FBPA_UC_DATA)
        hi = data >> 24
        temps[f] = (hi & 0x7F) if (hi & 0x80) == 0 else None

    def pkg_state(f):
        if (fbp_def >> f) & 1:
            return "defective"
        if (fbp_dis >> f) & 1:
            return "harvested"
        return "active"

    return {
        "bdf": bdf_s, "sm": sm, "pci_id": _pci_id(bdf_s),
        "gpc_dis": gpc_dis, "gpc_def": gpc_def,
        "nvl_dis": nvl_dis, "nvl_def": nvl_def,
        "tpc": tpc, "live": live, "live_n": len(live),
        "temps": temps, "pkg_state": pkg_state,
        "fbp_dis": fbp_dis, "fbp_def": fbp_def,
    }


def capacity(cs, gpf):
    """(live GiB, physical GiB) — from the decoded density when available,
    else from nvidia-smi's exposed total over the live FBPA count."""
    n = cs["live_n"]
    if gpf is not None:
        return n * gpf, FBPA_N * gpf
    mt = cs["sm"].get("mem_total")
    if mt and n:
        return mt / 1024.0, FBPA_N * (mt / n) / 1024.0
    return (mt / 1024.0 if mt else None), None


def stack_view(topo, cs):
    """Per-side draw lists (hbmmap.stack_pair shape + derived text).
    Dead boxes borrow the live stacks' vendor/density/ECC — one card
    ships one SKU, and all four live IDs agree — or '?' if no ID was
    read at all."""
    live = [s for s in topo["stacks"] if s["state"] == "live"]
    shared = None
    if live:
        m0 = live[0]["meta"]
        if m0 and all(
                s["meta"] and (s["meta"]["mfr"], s["meta"]["density"],
                               s["meta"]["ecc"])
                == (m0["mfr"], m0["density"], m0["ecc"])
                for s in live):
            shared = m0
    view = {"L": [], "R": []}
    for s in topo["stacks"]:
        if s["state"] == "live":
            m = s["meta"]
            ts = [t for t in (cs["temps"].get(f) for f in s["fbpas"])
                  if t is not None]
            temp = sum(ts) / len(ts) if ts else None
            state = "active"
        else:
            m = s["meta"] if s["meta"] is not None else shared
            temp, state = None, s["state"]
        view[s["side"]].append({
            "state": state, "temp": temp,
            "fbp": (s["fbps"][0], s["fbps"][-1]),
            "letter": s["letter"],
            "size": short_size(m), "hi": short_hi(m),
            "mfr": short_mfr(m), "ecc": short_ecc(m),
        })
    return view


def avg_temp(view):
    ts = [s["temp"] for row in view.values() for s in row
          if s["temp"] is not None]
    return sum(ts) / len(ts) if ts else None


# ---------------------------------------------------------------------------
# drawing (same geometry/primitives as hbmmap; the HBM boxes and the
# DEVICE_ID panel below the stats row are hbmmon's)

def stack_pair(cv, x, y, st, pkg_state):
    """Two FBP packages one above the other = one stack slot. Same shape
    as hbmmap.stack_pair; the ECC/size/vendor lines come from the live
    DEVICE_ID (or '?'):
    ┌ FBP00    16Gb ┐
    │ ECC       8Hi │
    ├─ A · 47°C  ───┤
    │ SK Hynix      │
    └ FBP02         ┘
    """
    w = hm.PAIR_W
    col = hm.STATE_COL[st["state"]]
    f0, f1 = st["fbp"]
    cv.box(x, y, w, hm.PAIR_H, col, tl="╭", tr="╮", bl="╰", br="╯")
    cv.put(x, y + 2, "├" + "─" * (w - 2) + "┤", col)

    def pkg_cap(cap_y, f):
        cv.put(x + 2, cap_y, f"FBP{f:02d}",
               "w" if pkg_state(f) == "active" else "d")

    pkg_cap(y, f0)
    pkg_cap(y + 4, f1)
    cv.put(x + 2, y + 1, st["ecc"], "d")
    cv.put(x + w - 2 - 3, y + 1, st["hi"], "d")
    cv.put(x + 2, y + 3, st["mfr"], "d")
    size = st["size"]
    sz_col = "w" if st["state"] == "active" else "d"
    cv.put(x + w - 2 - len(size), y, size, sz_col)
    t = f"{st['temp']:.0f}°C" if st["temp"] is not None else "--"
    spine = f" {st['letter']} · {t} "
    cv.put(x + (w - len(spine)) // 2, y + 2, spine, "w")


def draw_card(cv, cs, topo, hist=None):
    p = cv.put
    sm = cs["sm"]
    name = sm.get("name") or "GA100"
    view = stack_view(topo, cs)

    # blank row, then the card; identity line sits with the stats bars
    top = 1
    phase = time.monotonic() * 8.0

    # --- NVLink pads, top edge — one pad group per NVLINK fuse bit --------
    # Group count and state both come from the fuses (3 groups, all
    # fused off on this SKU). Fused-off links read as one "-" per group
    # (there is no traffic to measure); if any group were live the
    # fuse-derived count shows instead, since a rate would be invented.
    nvl = cs["nvl_dis"] | cs["nvl_def"]
    n_nvl = nvl.bit_length() or 3          # 3 = GA100 die constant
    n_act = sum(1 for i in range(n_nvl) if not ((nvl >> i) & 1))
    for i in range(n_nvl):
        col = ("r" if (cs["nvl_def"] >> i) & 1
               else "d" if (cs["nvl_dis"] >> i) & 1 else "g")
        p(8 + 12 * i, top, "▮" * 8, col)
    if n_act == 0:
        s = "/".join("-" * n_nvl)
        p(8 + 12 * n_nvl, top, f"↓{s} KB/s ↑{s} KB/s", "d")
    else:
        p(8 + 12 * n_nvl, top, f"{n_act}/{n_nvl} active",
          "g" if n_act == n_nvl else "d")

    # --- card body ----------------------------------------------------------
    cv.box(hm.BODY_X, top + 1, hm.BODY_W, hm.BODY_H, "d",
           tl="╭", tr="╮", bl="╰", br="╯", hs="═")

    # 8-pin EPS: gold chrome, each pin flickers g/y/r with power load.
    pin_x = hm.BODY_X + hm.BODY_W - 7
    pin_y = top + 2
    pwr_frac = hm.usage_frac(sm.get("pwr"), sm.get("pwr_lim"))
    pins = hm.usage_cells(8, pwr_frac, phase)
    p(pin_x, pin_y, "┌────┐", "d")
    p(pin_x, pin_y + 1, "│", "d")
    p(pin_x + 5, pin_y + 1, "│", "d")
    p(pin_x, pin_y + 2, "│", "d")
    p(pin_x + 5, pin_y + 2, "│", "d")
    p(pin_x, pin_y + 3, "└────┘", "d")
    for i in range(4):
        p(pin_x + 1 + i, pin_y + 1, pins[i][0], pins[i][1])
        p(pin_x + 1 + i, pin_y + 2, pins[i + 4][0], pins[i + 4][1])
    pwr = f"{hm._f(sm.get('pwr'))}/{hm._f(sm.get('pwr_lim'))} W"
    p(pin_x - 1 - len(pwr), pin_y + 1, pwr, "w")

    # mounting bracket (rear IO end, blower exhaust): full-height plate
    # with the screw tab above the PCB and a bottom lip by the slot
    br_bot = top + hm.BODY_H + 1          # PCIe finger row
    p(0, top, "┌─┐", "d")
    p(0, br_bot, "└─┘", "d")
    for r in range(top + 1, br_bot):
        p(0, r, "│", "d")
        p(2, r, "│", "d")
        p(1, r, "▒", "d")
    p(1, top + 1, "•", "d")        # screw hole in the top tab

    # GPU package sits left of center; each stack is two FBP chips stacked
    # vertically. Power 8-pin keeps the top-right corner.
    hb_left = hm.BODY_X + 4
    die_x = hb_left + hm.PAIR_W + hm.PKG_GAP
    die_y = top + 1 + (hm.BODY_H - hm.DIE_H) // 2
    hb_right = die_x + hm.DIE_W + hm.PKG_GAP
    hb_y = die_y + (hm.DIE_H - (hm.PAIR_H * 3 + 2)) // 2

    pkg_state = cs["pkg_state"]
    for k, st in enumerate(view["L"]):
        stack_pair(cv, hb_left, hb_y + k * (hm.PAIR_H + 1), st, pkg_state)
    for k, st in enumerate(view["R"]):
        stack_pair(cv, hb_right, hb_y + k * (hm.PAIR_H + 1), st, pkg_state)

    # --- GA100 die: 2×4 GPC grid of 3-row cells, SKU + GPU/MEM temp on rail
    cv.box(die_x, die_y, hm.DIE_W, hm.DIE_H, "d")

    def gpc_col(i):
        if (cs["gpc_def"] >> i) & 1:
            return "r"
        if (cs["gpc_dis"] >> i) & 1:
            return "d"
        return "g"

    def draw_gpc(gi, gx, gy):
        n_live = 8 - len(hm.bits_set(cs["tpc"][gi], 8))
        cv.box(gx, gy, hm.GPC_W, hm.GPC_H, gpc_col(gi),
               tl="╭", tr="╮", bl="╰", br="╯")
        p(gx + 2, gy, f" G{gi} ", "w")
        p(gx + 2, gy + 1, f"TPC {n_live}/8", "w")
        sms = f"{n_live * 2} SM"
        p(gx + hm.GPC_W - 2 - len(sms), gy + 1, sms, "w" if n_live else "d")

    gpc_gap = 2
    grid_w = hm.GPC_W * 2 + gpc_gap
    gx0 = die_x + (hm.DIE_W - grid_w) // 2
    # Die family is a tool constant (this register map is GA100); the
    # chip identity is sysfs-derived, so the label reads "GA100
    # 10de:20c2" on the 170HX instead of an assumed SKU string.
    pci_id = cs.get("pci_id")
    model = f"GA100 {pci_id}" if pci_id else "GA100"
    temp_s = (f"GPU {hm._f(sm.get('gpu_t'), 0)}°C  ")
    p(die_x + 2, die_y + 1, model, "B")
    p(die_x + hm.DIE_W - 2 - len(temp_s), die_y + 1, temp_s, "w")
    for i in range(4):
        gy = die_y + 2 + i * (hm.GPC_H + hm.GPC_VGAP)
        draw_gpc(i, gx0, gy)
        draw_gpc(i + 4, gx0 + hm.GPC_W + gpc_gap, gy)

    # --- PCIe x16 fingers, bottom edge near the bracket end ----------------
    py = br_bot
    short_n, gap, long_n, x0 = 6, 2, 30, 8    # the x16 physical edge
    n_fingers = short_n + long_n
    width = sm.get("pcie_w")
    max_w = int(sm.get("pcie_w_max") or 16)
    connected = n_fingers
    if width is not None:
        # lit prefix = current width against the card's max link width
        # (nvidia-smi), not an assumed x16
        connected = max(0, min(n_fingers,
                               int(round(n_fingers * int(width) / max_w))))
    bus_frac = hm.pcie_used_frac(sm)
    fingers = (hm.usage_cells(connected, bus_frac, phase * 1.4)
               + [("░", "d")] * (n_fingers - connected))
    for i, (ch, c) in enumerate(fingers[:short_n]):
        p(x0 + i, py, ch, c)
    for i, (ch, c) in enumerate(fingers[short_n:]):
        p(x0 + short_n + gap + i, py, ch, c)
    if sm.get("pcie_g") is not None and width is not None:
        link = f"{int(sm['pcie_g'])}.0 x{int(width)}"
        link_col = "G" if int(width) >= max_w else "y"
    else:
        link, link_col = "? x?", "d"
    rx, tx = sm.get("pcie_rx"), sm.get("pcie_tx")
    if rx is not None or tx is not None:
        link += f"  ↓{hm.fmt_tput(rx)} ↑{hm.fmt_tput(tx)}"
        if bus_frac is not None:
            link_col = hm.load_col(bus_frac)
    elif bus_frac is not None:
        link += f"  {int(round(bus_frac * 100))}%"
        link_col = hm.load_col(bus_frac)
    p(x0 + short_n + gap + long_n + 2, py, link, link_col)

    # blank line under the card, then identity + bars
    ly = br_bot + 2
    exposed, phys = capacity(cs, topo["gpf"])
    tpc_live = sum(8 - len(hm.bits_set(m, 8)) for m in cs["tpc"])
    sms = tpc_live * 2                              # GA100: 2 SM per TPC
    cores = sms * hm.CUDA_PER_SM
    p(2, ly, name, "C")
    x = 2 + len(name) + 2
    if exposed is not None:
        rest = (f"·  {exposed:.0f}/{phys:.0f} GiB  ·  "
                if phys is not None else
                f"·  {exposed:.0f} GiB  ·  ")
    else:
        rest = "·  ? GiB  ·  "
    rest += (f"{sms}/{hm.GA100_SM_FULL} SM  ·  "
             f"{cores}/{hm.GA100_SM_FULL * hm.CUDA_PER_SM} CUDA")
    p(x, ly, rest, "w")
    x += len(rest) + 2

    pcie_vals = list((hist or {}).get("pcie") or [])
    pwr_vals = list((hist or {}).get("pwr") or [])
    hbm_vals = list((hist or {}).get("hbm") or [])
    if not pcie_vals:
        rx, tx = sm.get("pcie_rx"), sm.get("pcie_tx")
        if rx is not None or tx is not None:
            pcie_vals = [max(rx or 0.0, tx or 0.0) / 1_000.0]
        elif sm.get("gpu_u") is not None:
            pcie_vals = [sm["gpu_u"]]
    if not pwr_vals and sm.get("pwr") is not None:
        pwr_vals = [sm["pwr"]]
    if not hbm_vals:
        a = avg_temp(view)
        if a is not None:
            hbm_vals = [a]

    pcie_cur = f" {pcie_vals[-1]:.0f} MB/s" if pcie_vals else ""
    pwr_cur = f" {pwr_vals[-1]:.0f} W" if pwr_vals else ""
    n_spark = (1 if len(pcie_vals) >= 2 else 0) + (1 if len(pwr_vals) >= 2 else 0)
    fixed = 0
    if pcie_vals:
        fixed += 5 + len(pcie_cur) + 2
    if pwr_vals:
        fixed += 4 + len(pwr_cur)
    spark_w = 8
    if n_spark:
        spark_w = max(8, min(32, (hm.W - 2 - x - fixed) // n_spark))

    def put_hist(x, label, vals, cur_s, cur_col):
        p(x, ly, label, "d")
        x += len(label)
        if len(vals) >= 2:
            sp = hm.sparkline(vals, width=spark_w)
            p(x, ly, sp, "c")
            x += len(sp)
            p(x, ly, cur_s, cur_col)
            x += len(cur_s)
        elif vals:
            s = cur_s[1:] if cur_s.startswith(" ") else cur_s
            p(x, ly, s, cur_col)
            x += len(s)
        return x + 2

    if pcie_vals:
        x = put_hist(x, "PCIe ", pcie_vals, pcie_cur, hm.load_col(bus_frac))
    if pwr_vals:
        x = put_hist(x, "PWR ", pwr_vals, pwr_cur, hm.load_col(pwr_frac))

    # --- nvtop-style stats panel ------------------------------------------
    sy = ly + 1
    x = 2
    p(x, sy, "GPU ", "d"); x += 4
    u = sm.get("gpu_u")
    if u is not None:
        col, b = hm.bar(u)
        p(x, sy, f"[{b}]", col); x += len(b) + 2
        p(x, sy, f" {int(u)}%", "w"); x += len(f" {int(u)}%")
    else:
        p(x, sy, "[?]", "d"); x += 5
    p(x + 2, sy, "MEM ", "d"); x += 6
    used, total = sm.get("mem_used"), sm.get("mem_total")
    # Occupancy (used/total), not nvidia-smi utilization.memory — that
    # field is memory-controller busy, so it can read 11% with 49/64 GiB
    # resident.
    if used is not None and total:
        mu = hm._clamp(100.0 * used / total, 0.0, 100.0)
    else:
        mu = sm.get("mem_u")
    if mu is not None:
        col, b = hm.bar(mu)
        p(x, sy, f"[{b}]", col); x += len(b) + 2
        pct_s = f" {int(round(mu))}%"
        p(x, sy, pct_s, "w"); x += len(pct_s)
        if used is not None and total is not None:
            g = f"  {used / 1024:.1f}/{total / 1024:.0f} GiB"
            p(x, sy, g, "w"); x += len(g)

    x += 2
    p(x, sy, "HBM avg ", "d")
    x += 8
    if len(hbm_vals) >= 2:
        room = max(8, hm.W - 2 - x - 5)
        sp = hm.sparkline(hbm_vals, width=min(32, room))
        p(x, sy, sp, "c")
        p(x + len(sp), sy, f" {hbm_vals[-1]:.0f}°C", "c")
    elif hbm_vals:
        p(x, sy, f"{hbm_vals[-1]:.0f}°C", "c")

    # --- DEVICE_ID panel (the point of hbmmon) -----------------------------
    y = sy + 2
    if topo["grouped"]:
        p(2, y, "── per-stack DEVICE_ID ─" * 20, "d")
        y += 1
        for s in topo["stacks"]:
            if s["ID"] is not None:
                d = s["meta"]
                dec = s.get("decoded", s["fbpas"])
                mark = (f"  ({len(dec)}/{len(s['fbpas'])} read)"
                        if len(dec) < len(s["fbpas"]) else "")
                p(2, y, f"  {s['letter']}  0x{s['ID']:021x}  "
                        f"FBPA {','.join(map(str, s['fbpas']))}  "
                        f"FBP {','.join(map(str, s['fbps']))}{mark}  "
                        f"serial 0x{d['serial']:09x}  "
                        f"{d['year']}-W{d['week']:02d}  "
                        f"CHAVAIL 0x{d['ch_avail']:02x}", "w")
            else:
                p(2, y, f"  {s['letter']}  (no DEVICE_ID — {s['state']}, "
                        f"no live FBPA)  FBP {','.join(map(str, s['fbps']))}",
                  "r" if s["state"] == "defective" else "d")
            y += 1
        m = topo["stacks"][0]["meta"]
        if m:
            p(2, y, f"  {MFR_NAMES.get(m['mfr'], '?')} · "
                    f"{DENSITY_NAMES.get(m['density'], '?')} · "
                    f"{'ECC cells' if m['ecc'] else 'no ECC cells'} · "
                    f"{ADDR_MODE_NAMES.get(m['addr_mode'], '?')} · "
                    f"MODEL 0x{m['model']:02x}"
                    + (f" · {exposed:.0f}/{phys:.0f} GiB live"
                       if exposed is not None and phys is not None else ""),
                    "w")
    else:
        live_list = cs.get("live") or set()
        dead = sorted({f // 2 for f in range(FBPA_N)
                       if f not in live_list})
        p(2, y, "── HBM identity: not read (read-only mode; FBP pairing "
          "assumed) " + "─" * 10, "d")
        y += 1
        p(2, y, f"  live FBPA:  {','.join(map(str, sorted(live_list)))}"
                f"   ({len(live_list)})"
                if live_list else "  live FBPA:  (none)", "w")
        y += 1
        p(2, y, f"  dead FBP:   {','.join(map(str, dead))}  "
                f"({len(dead) // 2} slot(s); pairing not recoverable at "
                "runtime)", "d")
        y += 1
        if exposed is not None:
            p(2, y, f"  capacity:   "
                    + (f"{exposed:.0f}/{phys:.0f} GiB "
                       if phys is not None else f"{exposed:.0f} GiB")
                    + " (nvidia-smi exposed / 24 FBPA slots)", "w")
            y += 1
        p(2, y, "  re-run without --read-only to read the per-stack "
                "DEVICE_ID", "d")
    return cv


def render_card(cs, topo, color, hist=None):
    cv = hm.Canvas(hm.W, 36)
    draw_card(cv, cs, topo, hist)
    lines = cv.render(color).split("\n")
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def header(color, live_ts=None, write_ok=False, mock=False):
    ts = f"   live @ {live_ts}" if live_ts else ""
    if mock:
        src = "MOCK · synthetic 170HX, no hardware"
    elif write_ok:
        src = "sweep"
    else:
        src = "read-only · BAR0 via /dev/gpcprobe"
    l1 = (f"hbmmon — HBM topology from live I1500 "
          f"DEVICE_ID{ts}   {src}")
    l2a = "legend:  "
    l2 = (f"{l2a}"
          f"\x1b[32m[active]\x1b[0m  "
          f"\x1b[2;37m[disabled]\x1b[0m  "
          f"\x1b[31m[defective]\x1b[0m")
    if not color:
        l2 = "legend:  [active]  [disabled]  [defective]"
    return "\n".join([l1, l2])


# ---------------------------------------------------------------------------

def print_id_table(topo):
    if not topo["grouped"]:
        print("  identity not read (read-only mode) — re-run without "
              "--read-only")
    for s in topo["stacks"]:
        if s["ID"] is not None:
            d = s["meta"]
            dec = s.get("decoded", s["fbpas"])
            mark = (f"  ({len(dec)}/{len(s['fbpas'])} read)"
                    if len(dec) < len(s["fbpas"]) else "")
            print(f"  {s['letter']}  0x{s['ID']:021x}  "
                  f"FBPA {','.join(map(str, s['fbpas']))}  "
                  f"FBP {','.join(map(str, s['fbps']))}{mark}  "
                  f"serial 0x{d['serial']:09x}  "
                  f"{d['year']}-W{d['week']:02d}  "
                  f"CHAVAIL 0x{d['ch_avail']:02x}  "
                  f"{MFR_NAMES.get(d['mfr'], '?')}  "
                  f"{DENSITY_NAMES.get(d['density'], '?')}  "
                  f"{'ECC' if d['ecc'] else 'no-ECC'}")
        else:
            print(f"  {s['letter']}  (no DEVICE_ID — {s['state']})  "
                  f"FBP {','.join(map(str, s['fbps']))}")


def mock_topo():
    """Measured 170HX identity snapshot (docs/hbm-1500-170hx.md §11.5)
    so --mock renders the grouped map without hardware."""
    IDs = (0x3a60092d4479203d9ff8a, 0x3a60092d4079203d9ff8a,
           0x3a60092d3a6d203d9ff8a, 0x3a60092d3c6d203d9ff8a)
    fbpas = ([0, 1, 4, 5], [6, 7, 10, 11],
             [14, 15, 18, 19], [16, 17, 20, 21])
    groups = {ID: list(fb) for ID, fb in zip(IDs, fbpas)}
    live = set(hm._MOCK_LIVE_FBPAS)
    stacks = derive_stacks(groups, live, hm._MOCK_FBP_DIS,
                           hm._MOCK_FBP_DEF)
    return {"stacks": stacks, "gpf": gib_per_fbpa(stacks[0]["meta"]),
            "grouped": True}


def _frame(cards, collect_one, topos, color, write_ok, mock,
           live_ts=None, hist=None):
    parts = [header(color, live_ts, write_ok, mock)]
    for i, c in enumerate(cards):
        cs = collect_one(c)
        if hist is not None:
            slot = hist.setdefault(c, {
                "hbm": deque(maxlen=48),
                "pcie": deque(maxlen=48),
                "pwr": deque(maxlen=48),
            })
            a = avg_temp(stack_view(topos[c], cs))
            if a is not None:
                slot["hbm"].append(a)
            rx, tx = cs["sm"].get("pcie_rx"), cs["sm"].get("pcie_tx")
            if rx is not None or tx is not None:
                slot["pcie"].append(max(rx or 0.0, tx or 0.0) / 1_000.0)
            else:
                u = cs["sm"].get("gpu_u")
                if u is not None:
                    slot["pcie"].append(u)
            pw = cs["sm"].get("pwr")
            if pw is not None:
                slot["pwr"].append(pw)
        if i:
            parts.append("")
        parts.append(f"══ Card {c} · {cs['bdf']} "
                     + "═" * max(0, 60 - len(cs['bdf'])))
        parts.append(render_card(cs, topos[c], color,
                                 None if hist is None else hist.get(c)))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description="170HX card monitor with the HBM stack map, identity "
                    "and capacity derived from the live per-FBPA "
                    "DEVICE_ID — same map as "
                    "hbmmap, nothing HBM hardcoded")
    ap.add_argument("--card", type=int, default=None,
                    help="module index (default: all cards; with --mock: 0)")
    ap.add_argument("--live", nargs="?", const=2.0, type=float, default=2.0,
                    metavar="SECS",
                    help="nvtop-style refresh interval (default: on, 2 s); "
                         "the DEVICE_ID sweep runs once at startup, not "
                         "per tick")
    ap.add_argument("--once", action="store_true",
                    help="single frame, no refresh loop (default: live)")
    ap.add_argument("--write-ok", action="store_true",
                    help="(compat, no-op) the startup DEVICE_ID sweep is "
                         "on by default; use --read-only to skip it")
    ap.add_argument("--read-only", action="store_true",
                    help="skip the startup DEVICE_ID sweep — BAR0 reads "
                         "only (identity lines read '?')")
    ap.add_argument("--ids", action="store_true",
                    help="print the per-stack DEVICE_ID table and exit")
    ap.add_argument("--mock", action="store_true",
                    help="synthetic 170HX snapshot — no /dev/gpcprobe, "
                         "no nvidia-smi (measured §11.5 identity)")
    ap.add_argument("--no-color", action="store_true", help="plain output")
    args = ap.parse_args()

    write_ok = not args.read_only            # sweep on by default
    live_interval = None if args.once else args.live

    color = (sys.stdout.isatty() and not args.no_color
             and not os.environ.get("NO_COLOR"))

    fd = None
    topos = {}
    if args.mock:
        cards = [0 if args.card is None else args.card]
        t_mono0 = time.monotonic()

        def collect_one(idx):
            t = (0.0 if live_interval is None
                 else (time.monotonic() - t_mono0))
            cs = hm.mock_collect(idx, t)
            cs["temps"] = hm.mock_temps(t)   # per-FBPA high bytes
            cs["pci_id"] = "10de:20c2"       # measured (vbios-re)
            return cs

        for c in cards:
            topos[c] = mock_topo()
        if args.ids:
            for c in cards:
                print_id_table(topos[c])
            return
    else:
        fd, info, count, fcntl, gp = hm._hw_open()
        if args.card is None:
            cards = list(range(count))
        elif not (0 <= args.card < count):
            os.close(fd)
            sys.exit(f"--card {args.card} out of range ({count} cards)")
        else:
            cards = [args.card]

        # --- startup: CFG1 map + (gated) DEVICE_ID sweep -------------------
        for c in cards:
            rd = hm._hw_rd(fd, fcntl, gp, c)
            bdf_s = hm._hw_bdf(info, c)
            live = hm.live_fbpas(rd)
            print(f"══ Card {c} · {bdf_s} · live FBPA {live}"
                  + ("" if write_ok
                     else "   (read-only: no DEVICE_ID read)"))
            if write_ok:
                def wr(off, val):
                    buf = bytearray(struct.pack("<III", c, off, val))
                    try:
                        fcntl.ioctl(fd, gp.GP_IOC_WRITE, buf)
                    except OSError as e:
                        sys.exit(f"write 0x{off:08x}<-0x{val:08x} refused "
                                 f"({e}); rebuild gpcprobe.ko with the "
                                 f"2026-09-19 extended I1500 allowlist?")
                groups = id_sweep(rd, wr, gp, live)
                if groups is None:
                    os.close(fd)
                    sys.exit(1)
            else:
                print(f"  (dry run: no writes; each of the {len(live)} live "
                      "FBPAs would arm MODE=0x52 + WIR 0x0F0E, capture 6 "
                      "words, restore+verify — re-run without --read-only "
                      "to read them)")
                groups = {}
            cs0 = collect(rd, bdf_s, {})
            if groups:
                stacks = derive_stacks(groups, set(live), cs0["fbp_dis"],
                                       cs0["fbp_def"])
                topos[c] = {"stacks": stacks,
                            "gpf": gib_per_fbpa(stacks[0]["meta"]),
                            "grouped": True}
            else:
                topos[c] = {"stacks": ro_stacks(set(live), cs0["fbp_dis"],
                                                cs0["fbp_def"]),
                            "gpf": None, "grouped": False}
            if args.ids:
                print_id_table(topos[c])
                if c == cards[-1]:
                    os.close(fd)
                    return

        smi_cache = {}
        smi_at = [0.0]

        def smi():
            now = time.monotonic()
            if smi_cache and now - smi_at[0] < 0.3:
                return smi_cache
            try:
                smi_cache.update(hm.smi_query())
                smi_at[0] = now
            except Exception:
                pass                   # keep last good values
            return smi_cache

        def collect_one(idx):
            return collect(hm._hw_rd(fd, fcntl, gp, idx),
                           hm._hw_bdf(info, idx), smi())

    try:
        if live_interval is None:
            print(_frame(cards, collect_one, topos, color,
                         write_ok, args.mock))
            return
        interval = max(0.5, live_interval)
        hist = {}
        while True:
            tick = time.time()
            sys.stdout.write(
                "\x1b[H\x1b[2J"
                + _frame(cards, collect_one, topos, color,
                         write_ok, args.mock,
                         time.strftime("%H:%M:%S"), hist)
                + "\n")
            sys.stdout.flush()
            time.sleep(max(0.0, interval - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    main()
