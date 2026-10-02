"""hbmmap — shared card-map library for hbmmon (CMP 170HX, GA100/A100).

No CLI of its own — the tool is hbmmon.py. This module provides the
drawing canvas, the die/card geometry constants, the nvidia-smi
collection, the --mock synthetic snapshot and the /dev/gpcprobe
hardware access that hbmmon builds its map on.

READ-ONLY: BAR0 reads only (GP_IOC_READ32) plus the nvidia-smi CSV
query. The only writes the card sees come from hbmmon's startup
DEVICE_ID sweep, gated in gpc_probe.wdr_arm_capture and allowlisted in
gpcprobe.c (I1500 INSTR/MODE control registers only).

Colors: green = active/healthy, dark = disabled (fused off, healthy),
red = defective.

FBP -> stack map: MEASURED, non-contiguous (170hx repo
docs/hbm-1500-170hx.md §4.1, 2026-09-19 unicast discovery). The pairing
is self-checking: on this card every stack's four FBPAs are either all
present (unicast CFG1 responds) or all absent (0xBADFxxxx). hbmmon
derives the map from hardware and uses this table only for --mock.
"""

import math
import os
import struct
import subprocess
import sys

# --------------------------------------------------------------------------
# FBP -> stack map — MEASURED, non-contiguous (§4.1).
# Stack = 2 FBP = 4 FBPA (2 each) = 8 channels = 16 GiB.
#   A-D: live (all 4 FBPAs present in the unicast aperture)
#   E:   floorswept — OPT_FBPA_DISABLE set, DEFECTIVE clear (harvested)
#   F:   defective — both set
# Column placement in the drawing (6 per side, FBP index order).
STACKS = (
    ("A", (0, 2)),    # FBPA 0,1,4,5
    ("B", (3, 5)),    # FBPA 6,7,10,11
    ("C", (7, 9)),    # FBPA 14,15,18,19
    ("D", (8, 10)),   # FBPA 16,17,20,21
    ("E", (1, 4)),    # FBPA 2,3,8,9    — floorswept (healthy)
    ("F", (6, 11)),   # FBPA 12,13,22,23 — defective
)

OFF_GPC_DIS, OFF_GPC_DEF = 0x00820350, 0x008205C4
OFF_FBP_DIS, OFF_FBP_DEF = 0x00820364, 0x008205CC
OFF_NVL_DIS, OFF_NVL_DEF = 0x00820684, 0x0082068C
OFF_PCIE_DIS = 0x00820394
OFF_GEN3_DIS = 0x00820580
OFF_TPC_STATUS = 0x00820C38          # + i*4, one dword per GPC (remove-only)

# Per-FBPA unicast I1500 (same numbers as gpc_probe.FBPA_UC_*)
FBPA_UC_BASE, FBPA_UC_STRIDE = 0x00900000, 0x00004000
FBPA_UC_COUNT = 24
FBPA_UC_CFG1, FBPA_UC_DATA = 0x0204, 0x3CBC


def bits_set(mask, width):
    return [i for i in range(width) if mask & (1 << i)]


def live_fbpas(rd):
    """FBPAs whose unicast aperture responds (cfg1 high half != 0xBADF)."""
    live = []
    for f in range(FBPA_UC_COUNT):
        cfg1 = rd(FBPA_UC_BASE + f * FBPA_UC_STRIDE + FBPA_UC_CFG1)
        if (cfg1 & 0xFFFF0000) != 0xBADF0000:
            live.append(f)
    return live

PAL = {
    "g": "\x1b[32m", "G": "\x1b[1;32m",
    "y": "\x1b[33m", "Y": "\x1b[1;33m",
    "r": "\x1b[31m", "R": "\x1b[1;31m",
    "d": "\x1b[2;37m",
    "c": "\x1b[36m", "C": "\x1b[1;36m",
    "b": "\x1b[34m", "B": "\x1b[1;34m",
    "m": "\x1b[35m", "M": "\x1b[1;35m",
    "w": "\x1b[1;37m",
    "K": "\x1b[0m",
}
STATE_COL = {"active": "g", "harvested": "d", "defective": "r"}
SPARK = "▁▂▃▄▅▆▇█"


class Canvas:
    """Char grid with per-cell colors; ANSI runs collapsed on render."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ch = [[" "] * w for _ in range(h)]
        self.cl = [[None] * w for _ in range(h)]

    def put(self, x, y, s, c=None):
        if not (0 <= y < self.h):
            return
        for i, k in enumerate(s):
            xx = x + i
            if 0 <= xx < self.w:
                self.ch[y][xx] = k
                if c:
                    self.cl[y][xx] = c

    def box(self, x, y, w, h, c, tl="┌", tr="┐", bl="└", br="┘",
            hs="─", vs="│"):
        self.put(x, y, tl + hs * (w - 2) + tr, c)
        self.put(x, y + h - 1, bl + hs * (w - 2) + br, c)
        for r in range(1, h - 1):
            self.put(x, y + r, vs, c)
            self.put(x + w - 1, y + r, vs, c)

    def render(self, color):
        out = []
        for y in range(self.h):
            row, cur = [], None
            for x in range(self.w):
                k, c = self.ch[y][x], self.cl[y][x]
                if c != cur:
                    if cur is not None and color:
                        row.append(PAL["K"])
                    if c and color:
                        row.append(PAL[c])
                    cur = c
                row.append(k)
            if cur is not None and color:
                row.append(PAL["K"])
            out.append("".join(row).rstrip())
        return "\n".join(out)


# ---------------------------------------------------------------------------
# data collection (read-only)

SMI_Q = ("index,pci.bus_id,name,temperature.gpu,temperature.memory,"
         "utilization.gpu,utilization.memory,memory.used,memory.total,"
         "power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current,"
         "pcie.link.width.max")


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def _tput_kb(raw):
    """Parse an nvidia-smi throughput field ('1250 KB/s', '1 MB/s', 'N/A')."""
    raw = (raw or "").strip()
    if not raw or raw.upper() in ("N/A", "[N/A]", "NOT SUPPORTED"):
        return None
    parts = raw.replace("/s", "").split()
    val = _num(parts[0].replace(",", ""))
    if val is None:
        return None
    unit = (parts[1] if len(parts) > 1 else "KB").upper()
    if unit.startswith("G"):
        return val * 1_000_000.0
    if unit.startswith("M"):
        return val * 1_000.0
    return val


def _bdf_norm(s):
    """00000000:06:00.0 / 0000:06:00.0 → 0000:06:00.0."""
    s = (s or "").lower().strip()
    if ":" not in s:
        return s
    dom, rest = s.split(":", 1)
    return f"{dom[-4:]}:{rest}"


def _parse_q_tput(text):
    """Parse Tx/Rx Throughput lines from `nvidia-smi -q` text, keyed by BDF."""
    out, bdf = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("GPU "):
            parts = s.split()
            # BDF is domain:bus:dev.fn — not Utilization's "GPU : 4 %"
            if len(parts) >= 2 and parts[1].count(":") >= 2:
                bdf = _bdf_norm(parts[1])
                out.setdefault(bdf, {})
        elif bdf and s.startswith("Tx Throughput"):
            out[bdf]["pcie_tx"] = _tput_kb(s.split(":", 1)[-1])
        elif bdf and s.startswith("Rx Throughput"):
            out[bdf]["pcie_rx"] = _tput_kb(s.split(":", 1)[-1])
    return out


def _tput_field(s):
    """CSV throughput cell: bare KB/s, or a unit-bearing string, or N/A."""
    s = (s or "").strip()
    if not s:
        return None
    if any(c.isalpha() for c in s):
        return _tput_kb(s)
    return _num(s)


def _has_tput(recs):
    return any(v.get("pcie_rx") is not None or v.get("pcie_tx") is not None
               for v in recs.values())


def smi_dmon_tput():
    """Index → {pcie_rx, pcie_tx} in KB/s from `nvidia-smi dmon -c 1 -s t`."""
    p = subprocess.run(["nvidia-smi", "dmon", "-c", "1", "-s", "t"],
                       capture_output=True, text=True, timeout=15)
    if p.returncode:
        return {}
    by_idx = {}
    for line in p.stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        rx_mb, tx_mb = _num(parts[1]), _num(parts[2])
        by_idx[idx] = {
            "pcie_rx": None if rx_mb is None else rx_mb * 1_000.0,
            "pcie_tx": None if tx_mb is None else tx_mb * 1_000.0,
        }
    return by_idx


def smi_pcie_tput():
    """Per-BDF Rx/Tx throughput in KB/s. Hardware path (not --mock).

    `-d THROUGHPUT` is not a valid nvidia-smi display type, so we never
    use it. Prefer the CSV query fields, then `-q -d PCI`, then full `-q`.
    """
    for args in (
        ["nvidia-smi", "-q", "-d", "PCI"],
        ["nvidia-smi", "-q"],
    ):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=12)
        except Exception:
            continue
        if p.returncode:
            continue
        parsed = _parse_q_tput(p.stdout)
        if _has_tput(parsed):
            return parsed
    try:
        p = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=pci.bus_id,pcie.rx_throughput,pcie.tx_throughput",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        p = None
    if p is not None and p.returncode == 0:
        out = {}
        for line in p.stdout.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 3:
                continue
            out[_bdf_norm(f[0])] = {
                "pcie_rx": _tput_field(f[1]),
                "pcie_tx": _tput_field(f[2]),
            }
        if _has_tput(out):
            return out
    return {}


def smi_query():
    """One nvidia-smi CSV call for all fields, keyed by lowercase BDF."""
    p = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_Q}",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10)
    if p.returncode:
        return {}
    out = {}
    for line in p.stdout.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != 14:
            continue
        bdf = _bdf_norm(f[1])
        try:
            idx = int(float(f[0]))
        except ValueError:
            idx = None
        out[bdf] = {
            "idx": idx,
            "name": f[2],
            "gpu_t": _num(f[3]), "mem_t": _num(f[4]),
            "gpu_u": _num(f[5]), "mem_u": _num(f[6]),
            "mem_used": _num(f[7]), "mem_total": _num(f[8]),
            "pwr": _num(f[9]), "pwr_lim": _num(f[10]),
            "pcie_g": _num(f[11]), "pcie_w": _num(f[12]),
            "pcie_w_max": _num(f[13]),
        }
    try:
        tput = smi_pcie_tput()
    except Exception:
        tput = {}
    for bdf, rec in out.items():
        rec.update(tput.get(bdf, {}))
        if rec.get("pcie_rx") is None and rec.get("pcie_tx") is None:
            suf = bdf.split(":", 1)[-1]
            for k, v in tput.items():
                if k.count(":") >= 2 and k.endswith(suf):
                    rec.update(v)
                    break
    if not _has_tput(out):
        try:
            by_idx = smi_dmon_tput()
        except Exception:
            by_idx = {}
        for rec in out.values():
            extra = by_idx.get(rec.get("idx"))
            if extra:
                rec.update(extra)
    return out


def collect(rd, bdf_s, smi):
    """Read one card's fuses + unicast stack temps. No writes, ever.

    `rd(off)` is a BAR0 dword reader (hardware ioctl or --mock).
    """
    sm = smi.get(bdf_s.lower(), {})

    gpc_dis = rd(OFF_GPC_DIS)
    gpc_def = rd(OFF_GPC_DEF)
    fbp_dis = rd(OFF_FBP_DIS)
    fbp_def = rd(OFF_FBP_DEF)
    nvl_dis = rd(OFF_NVL_DIS)
    nvl_def = rd(OFF_NVL_DEF)
    pcie_dis = rd(OFF_PCIE_DIS)
    gen3_dis = rd(OFF_GEN3_DIS) & 1
    tpc = [rd(OFF_TPC_STATUS + 4 * i) & 0xFF for i in range(8)]
    live = set(live_fbpas(rd))

    stacks = {}
    for letter, (f0, f1) in STACKS:
        fbpas = (2 * f0, 2 * f0 + 1, 2 * f1, 2 * f1 + 1)
        live_f = [f for f in fbpas if f in live]
        if live_f:
            state = "active"
            data = rd(FBPA_UC_BASE + live_f[0] * FBPA_UC_STRIDE +
                      FBPA_UC_DATA)
            hi = data >> 24
            temp = (hi & 0x7F) if (hi & 0x80) == 0 else None
        else:
            state = ("defective"
                     if ((fbp_def >> f0) & 1) or ((fbp_def >> f1) & 1)
                     else "harvested")
            temp = None
        stacks[letter] = {"state": state, "temp": temp, "fbp": (f0, f1)}

    def pkg_state(f):
        if (fbp_def >> f) & 1:
            return "defective"
        if (fbp_dis >> f) & 1:
            return "harvested"
        return "active"

    return {
        "bdf": bdf_s, "sm": sm,
        "gpc_dis": gpc_dis, "gpc_def": gpc_def,
        "nvl_dis": nvl_dis, "nvl_def": nvl_def,
        "pcie_lanes_on": 16 - len(bits_set(pcie_dis, 16)),
        "gen3_dis": gen3_dis, "tpc": tpc,
        "stacks": stacks, "live_n": len(live), "pkg_state": pkg_state,
    }


# ---------------------------------------------------------------------------
# --mock: synthetic 170HX (card 06 snapshot) so the drawing can be iterated
# locally with no kernel module and no box.

# GPC 0,1 harvested; GPC 3 defective. FBP 1,4 harvested (stack E); FBP 6,11
# defective (stack F). NVLink all fused off. 16 live FBPAs / 64 GiB.
_MOCK_GPC_DIS, _MOCK_GPC_DEF = 0x0B, 0x08
_MOCK_FBP_DIS = (1 << 1) | (1 << 4) | (1 << 6) | (1 << 11)
_MOCK_FBP_DEF = (1 << 6) | (1 << 11)
_MOCK_NVL_DIS, _MOCK_NVL_DEF = 0x07, 0x00
_MOCK_LIVE_FBPAS = frozenset((
    0, 1, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21,
))
_MOCK_CFG1_LIVE = 0x02779000          # A100-80GB HBM2E config
_MOCK_STACK_TEMP = {"A": 47, "B": 46, "C": 46, "D": 45}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def mock_temps(t):
    """Per-FBPA TEMPERATURE WDR high-bytes. Exact snapshot at t=0; wobble in --live."""
    out = {}
    for letter, (f0, f1) in STACKS:
        base = _MOCK_STACK_TEMP.get(letter)
        if base is None:
            continue
        if t:
            phase = 0.7 * (ord(letter) - ord("A"))
            val = int(round(base + 1.6 * math.sin(t / 5.0 + phase)))
        else:
            val = base
        for f in (2 * f0, 2 * f0 + 1, 2 * f1, 2 * f1 + 1):
            out[f] = val
    return out


def mock_rd(temps):
    """BAR0 dword reader for the synthetic fuse + unicast map."""
    tpc = [0xFF if (_MOCK_GPC_DIS >> i) & 1 else 0x01 for i in range(8)]

    def rd(off):
        if off == OFF_GPC_DIS:
            return _MOCK_GPC_DIS
        if off == OFF_GPC_DEF:
            return _MOCK_GPC_DEF
        if off == OFF_FBP_DIS:
            return _MOCK_FBP_DIS
        if off == OFF_FBP_DEF:
            return _MOCK_FBP_DEF
        if off == OFF_NVL_DIS:
            return _MOCK_NVL_DIS
        if off == OFF_NVL_DEF:
            return _MOCK_NVL_DEF
        if off == OFF_PCIE_DIS:
            return 0
        if off == OFF_GEN3_DIS:
            return 0
        if OFF_TPC_STATUS <= off < OFF_TPC_STATUS + 32:
            return tpc[(off - OFF_TPC_STATUS) // 4]
        if off >= FBPA_UC_BASE:
            f, intra = divmod(off - FBPA_UC_BASE, FBPA_UC_STRIDE)
            if intra == FBPA_UC_CFG1:
                return _MOCK_CFG1_LIVE if f in _MOCK_LIVE_FBPAS else 0xBADF0000
            if intra == FBPA_UC_DATA:
                t = temps.get(f)
                return 0 if t is None else (t & 0x7F) << 24
        return 0

    return rd


def mock_smi(bdf_s, t):
    """nvidia-smi-shaped dict; util/power/GPU temp drift in --live."""
    gpu_u, mem_u, pwr = 73.0, 41.0, 218.0
    gpu_t, mem_t, mem_used = 62.0, 50.0, 26880.0
    if t:
        gpu_u = _clamp(73 + 18 * math.sin(t / 3.0), 0, 100)
        mem_u = _clamp(41 + 12 * math.sin(t / 4.5 + 1.0), 0, 100)
        pwr = _clamp(218 + 22 * math.sin(t / 4.0), 80, 250)
        gpu_t = _clamp(62 + 6 * math.sin(t / 6.0), 30, 90)
        mem_t = _clamp(50 + 2 * math.sin(t / 7.0 + 0.4), 30, 90)
        mem_used = 26880 + 4096 * math.sin(t / 8.0)
    return {
        bdf_s.lower(): {
            "name": "NVIDIA CMP 170HX",
            "gpu_t": gpu_t, "mem_t": mem_t,
            "gpu_u": gpu_u, "mem_u": mem_u,
            "mem_used": mem_used, "mem_total": 65536,
            "pwr": pwr, "pwr_lim": 250,
            "pcie_g": 4, "pcie_w": 16, "pcie_w_max": 16,
            # KB/s — modest host traffic, not the 31.5 GB/s link peak
            "pcie_rx": 180_000.0 * (gpu_u / 100.0),
            "pcie_tx": 40_000.0 * (gpu_u / 100.0),
        }
    }


def mock_collect(idx, t=0.0):
    """One synthetic card. `idx` only affects the banner/BDF."""
    bdf_s = f"0000:{6 + idx:02x}:00.0"
    return collect(mock_rd(mock_temps(t)), bdf_s, mock_smi(bdf_s, t))


# ---------------------------------------------------------------------------
# drawing

W = 124
BODY_X, BODY_W, BODY_H = 3, 118, 20
# Full GA100: 8 GPC × 8 TPC × 2 SM. Datacenter Ampere SM = 64 FP32 CUDA
# cores (GA10x consumer SMs count 128 because INT32 pipes also do FP32).
GA100_SM_FULL = 128
CUDA_PER_SM = 64
# Two FBP packages stacked = one 16 GiB controller slot. Three pairs
# per side with a 1-row gap; GPC cells are 3-row boxes in a 2×4 grid.
PAIR_W, PAIR_H = 18, 5
DIE_W, DIE_H = 46, 18
GPC_W, GPC_H = 20, 3
GPC_VGAP = 1         # blank row between GPC cells so they don't fuse
PKG_GAP = 2          # columns between an HBM pair and the die


def _f(x, nd=1, unit=""):
    return f"{x:.{nd}f}{unit}" if x is not None else "?"


def bar(pct, width=22):
    n = int(round(pct / 100.0 * width))
    col = "g" if pct < 70 else ("y" if pct < 90 else "r")
    return col, "█" * n + "░" * (width - n)


def pcie_gt(gen):
    """Lane signaling rate in GT/s for a PCIe generation."""
    if gen is None:
        return None
    return {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}.get(int(gen))


def pcie_gbps(gen, width):
    """Unidirectional payload GB/s from link gen × lane width."""
    gt = pcie_gt(gen)
    if gt is None or width is None or int(width) <= 0:
        return None
    enc = 8 / 10 if int(gen) <= 2 else 128 / 130
    return gt * int(width) * enc / 8.0


def pcie_used_frac(sm):
    """Hotter of Rx/Tx vs link capacity. Falls back to GPU util."""
    peak = pcie_gbps(sm.get("pcie_g"), sm.get("pcie_w"))
    rx, tx = sm.get("pcie_rx"), sm.get("pcie_tx")
    if peak and (rx is not None or tx is not None):
        return usage_frac(max(rx or 0.0, tx or 0.0), peak * 1_000_000.0)
    return usage_frac(sm.get("gpu_u"), 100.0)


def fmt_tput(kb):
    """KB/s → a short human rate."""
    if kb is None:
        return "?"
    if kb >= 1_000_000:
        return f"{kb / 1_000_000:.1f} GB/s"
    if kb >= 1_000:
        mb = kb / 1_000
        return f"{mb:.1f} MB/s" if mb < 10 else f"{mb:.0f} MB/s"
    return f"{kb:.0f} KB/s"


def usage_frac(num, den):
    if num is None or den is None or den <= 0:
        return None
    return _clamp(num / den, 0.0, 1.0)


def load_col(frac):
    """Traffic-light for a 0..1 usage fraction. None / idle chrome is gold."""
    if frac is None:
        return "Y"
    if frac < 0.70:
        return "g"
    if frac < 0.90:
        return "y"
    return "r"


def usage_cells(n, frac, phase):
    """Gold ▮ chrome. Live pins flicker g/y/r around the current load."""
    if frac is None:
        return [("▮", "Y")] * n
    out = []
    for i in range(n):
        if frac < (i + 0.5) / n:
            out.append(("▮", "Y"))
            continue
        local = _clamp(frac + 0.18 * math.sin(phase + i * 1.15), 0.0, 1.0)
        out.append(("▮", load_col(local)))
    return out


def sparkline(hist, lo=None, hi=None, width=32):
    vals = list(hist)[-width:]
    if not vals:
        return ""
    if lo is None or hi is None:
        lo, hi = min(vals), max(vals)
        if hi - lo < 4:
            lo, hi = max(0.0, hi - 4), hi
    if hi <= lo:
        hi = lo + 1.0
    body = "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.99))]
                   for v in vals)
    return " " * (width - len(body)) + body


# ---------------------------------------------------------------------------
# /dev/gpcprobe hardware access (only used off --mock)

def _hw_open():
    """Load gpc_probe + open /dev/gpcprobe. Only used off --mock."""
    try:
        import fcntl
        import gpc_probe as gp
    except ImportError as e:
        sys.exit(f"hardware path needs gpc_probe + fcntl ({e}); "
                 f"use --mock to render a synthetic 170HX map")
    try:
        fd = os.open("/dev/gpcprobe", os.O_RDWR)
    except OSError as e:
        sys.exit(f"cannot open /dev/gpcprobe ({e}); run: make && sudo insmod "
                 f"gpcprobe.ko (or pass --mock)")
    info = bytearray(4 + 16 * 4)
    fcntl.ioctl(fd, gp.GP_IOC_INFO, info, True)
    count = struct.unpack_from("<I", info)[0]
    if not count:
        os.close(fd)
        sys.exit("no NVIDIA GPUs seen by gpcprobe")
    return fd, info, count, fcntl, gp


def _hw_rd(fd, fcntl, gp, idx):
    def rd(off):
        buf = bytearray(struct.pack("<III", idx, off, 0))
        fcntl.ioctl(fd, gp.GP_IOC_READ32, buf)
        return struct.unpack_from("<I", buf, 8)[0]
    return rd


def _hw_bdf(info, idx):
    bdf = struct.unpack_from("<I", info, 4 + 4 * idx)[0]
    bus, devfn = (bdf >> 8) & 0xFF, bdf & 0xFF
    return f"0000:{bus:02x}:{devfn >> 3:02x}.{devfn & 7}"
