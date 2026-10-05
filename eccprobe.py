#!/usr/bin/env python3
"""eccprobe — CMP 170HX (GA100) ECC unit state, from BAR0 registers + the
I1500 HBM debug bridge. No nvidia-smi.

Sections per card:
  SRAM ECC — the 18 SRAM units of the three NV_FUSE_FEATURE_OVERRIDE_ECC*
            registers (0x82380c: 5 of 6 nibbles, 0x823810: 11 pairs,
            0x82382c: 2 pairs), decoded per unit (override + value) and
            cross-checked against FEATURE_READOUT 0x823814, which lights
            one bit per unit the GSP successfully initialized.
  DRAM ECC — the 6th 0x82380c unit (readout bit 16), the FUSE_ECC_EN
            0x820228 fuse shadow, and the FBPA sideband datapath arm:
            per-partition 0x00900000+pt*0x4000 offsets +0x470/+0x1004/
            +0x33C/+0x84 (bit masks 0x41/0x200/0x3/0x1) + globals
            0x17E318/0x17E21C bit 0 + broadcast FBPA_ECC_CTRL 0x9A0470.
  HBM mode — MODE_REGISTER_DUMP (WIR 0x0010, channel 0) over I1500:
            MR4 OP[1:0] (JESD235D Table 15) says what the dies run in:
              00 = DM on, ECC off      -> "DM"
              10 = DM off, ECC off     -> "NONE"
              11 = DM off, ECC on      -> "ECC"
            (01 reserved). The die stores host-computed check bits in
            DM-pin cells; OP=11 is the only mode where they come back
            on reads.

Writes: ONLY the I1500 INSTR/MODE control registers, which the gpcprobe
kernel module hard-allowlists (the DATA latch is not host-writable — a
DATA-latch write wedges the bridge card-wide, 2026-09-25 canaries).
The HBM query arms the read-only 10h WDR with the signed-off §11.8
recipe (MODE=0x52 stream arm, 5-word capture, ISR-race re-arms), diffs
the HBM sentinels (FBPA_NUM_ACTIVE / FBPA_TRAINING) and dmesg Xids, and
aborts on any change. `--read-only` skips the HBM query (zero writes).

Usage:
  ./eccprobe.py                 # all cards, full probe
  ./eccprobe.py --card 1        # one gpcprobe card index
  ./eccprobe.py --read-only     # no I1500 arm (HBM mode not queried)
  ./eccprobe.py --trace         # dump raw capture words per attempt
"""
import argparse
import fcntl
import os
import struct
import sys
import time

from gpc_probe import (
    GP_IOC_INFO, GP_IOC_READ32, GP_IOC_WRITE,
    wdr_arm_capture, WDR_MODE_STREAM,
    hbm_sentinels, xid_since,
    FBPA_UC_BASE, FBPA_UC_STRIDE, FBPA_UC_COUNT,
)

DEV = "/dev/gpcprobe"

# ---------------------------------------------------------------- unit map
# Published vendor field map (docs/ecc-unlock-plan.md) + readout-bit map
# confirmed live 2026-09-25..27 (phase1/all9/bisect boots).
#
# 0x82380c: 6 units x 4-bit nibbles [value @4i (mode bits 4i..4i+2),
#           override @4i+3]. Stock 0x00888888 = all ovr=1, val=0.
# 0x823810: 11 units x 2-bit pairs [value @2i, override @2i+1].
#           Stock 0x002AAAAA.
# 0x82382c: 2 units x 2-bit pairs [value @2i, override @2i+1].
#           Stock 0x0000000A.
# readout bit = FEATURE_READOUT 0x823814 bit the GSP lights for a
# successfully-initialized unit; None = no readout bit (presence is the
# surviving-boot witness).
ECC0 = [  # (nibble idx, name, robit, known)
    (0, "SM_LRF",            12, "present"),
    (1, "SM_L1_DATA",        13, "present"),
    (2, "SM_L1_TAG",         14, "present"),
    (3, "LTC",               15, "present"),
    (4, "DRAM",              16, "present"),
    (5, "SM_CBU",            17, "present"),
]
ECC1 = [  # (data bit, name, robit, known)
    (0,  "SM_L0_ICACHE",         18, "present"),
    (2,  "SM_L1_ICACHE",         19, "present"),
    (4,  "SM_L1_MISS_LAT_FIFO",  23, "present"),
    (6,  "SM_PIXRF",             24, "ABSENT"),
    (8,  "LTC_L2_DCACHE_TAG",    25, "ABSENT"),
    (10, "FECS_FALCON",          26, "present"),
    (12, "GPCCS_FALCON",         27, "present"),
    (14, "PMU_FALCON",           None, "present"),
    (16, "GCC_L1_5_CACHE",       29, "present"),
    (18, "HUBMMU",               30, "present"),
    (20, "GPCMMU",               31, "present"),
]
ECC2 = [  # (data bit, name, known)
    (0, "LTC_CBC", "present"),
    (2, "SM_URF",  "present"),
]

# stock values — the SKU's forced-off baseline (all ovr=1, val=0)
STOCK = {0x0082380C: 0x00888888, 0x00823810: 0x002AAAAA,
         0x0082382C: 0x0000000A}

R_FEAT_ECC0 = 0x0082380C
R_FEAT_ECC1 = 0x00823810
R_FEAT_RO   = 0x00823814
R_FEAT_ECC2 = 0x0082382C
R_FUSE_ECC  = 0x00820228    # FUSE_ECC_EN (OTP shadow; fused off on this SKU)
R_FBPA_CTRL = 0x009A0470    # broadcast FBPA_ECC_CTRL (MASTER_EN)

# the pre-GSP Booter's sideband-ECC arm (ECC_ARM block): RMW-set bits
ARM_OFFS  = (0x470, 0x1004, 0x33C, 0x84)
ARM_BITS  = (0x41, 0x200, 0x3, 0x1)
ARM_GLOBALS = (0x17E318, 0x17E21C)
ARM_GLOBAL_BIT = 0x1


class Card:
    def __init__(self, fd, idx, bdf):
        self.fd, self.idx, self.bdf = fd, idx, bdf

    def rd(self, off):
        buf = bytearray(struct.pack("<III", self.idx, off, 0))
        fcntl.ioctl(self.fd, GP_IOC_READ32, buf)
        return struct.unpack_from("<I", buf, 8)[0]

    def wr(self, off, val):
        buf = bytearray(struct.pack("<III", self.idx, off, val))
        fcntl.ioctl(self.fd, GP_IOC_WRITE, buf)

    @property
    def name(self):
        bus = (self.bdf >> 8) & 0xFF
        dev = (self.bdf >> 3) & 0x1F
        fn = self.bdf & 7
        return f"{bus:02x}:{dev:02x}.{fn}"


def live_fbpas(rd):
    """Self-describing live map: a floorswept partition's unicast CFG1
    (0x00900204+pt*0x4000) reads 0xBADFxxxx (same test the Booter's
    arm uses for its poison-skip)."""
    live = []
    for pt in range(FBPA_UC_COUNT):
        cfg1 = rd(0x00900204 + pt * FBPA_UC_STRIDE)
        if (cfg1 >> 8) == 0xBADF20 or (cfg1 >> 12) == 0xBADF:
            continue
        live.append(pt)
    return live


# ------------------------------------------------- I1500 MR-dump (10h) WDR
def mrdump_align(data_words):
    """Align a 0x10 read window to the stream's chunk cycle.

    Ported verbatim from the 170hx repo gpc_probe.py (first live run
    2026-09-20, §11.9): the payload is a 3-chunk / 96-bit cycle
    (MR0-11); 0x03000000 is the absorbing idle value in the gap between
    bursts; 0xT0000000 (low 24 zero, high byte >= 0x20) is the per-stack
    temperature word after the Falcon ISR re-armed the WIR to 0x0F.
    The cycle ORDER is solved from observed adjacent transitions; the
    head anchor pins position 0 to the MR0 chunk (no live window ever
    began mid-burst).
    """
    import itertools
    IDLE = 0x03000000

    def is_temp(wd):
        return wd & 0xFFFFFF == 0 and (wd >> 24) & 0x7F >= 0x20

    data = [wd for wd in data_words if wd != IDLE and not is_temp(wd)]
    if not data:
        return None, "no cycle chunks (idle/temp only — gap or ISR)"
    vals = list(dict.fromkeys(data))
    if len(vals) == 1:
        return vals, f"single chunk 0x{vals[0]:08x} (cycle length/pos unknown)"
    edges = set(zip(data, data[1:]))
    good = []
    for perm in itertools.permutations(vals):
        pos = {v: i for i, v in enumerate(perm)}
        if all(pos[b] == (pos[a] + 1) % len(perm) for a, b in edges):
            good.append(perm)
    if not good:
        return None, (f"no closed cycle over "
                      f"{['0x%08x' % v for v in vals]} "
                      f"(edges {sorted('0x%08x/%08x' % e for e in edges)})")
    prev = None
    head = data[0]
    for wd in data_words:
        if wd != IDLE and not is_temp(wd) and (
                prev is None or prev == IDLE or is_temp(prev)):
            head = wd
            break
        prev = wd
    anchored = [p for p in good if p[0] == head]
    if len(anchored) == 1:
        return anchored[0], (f"{len(vals)}-chunk cycle, "
                             + ("order+head anchored" if len(good) > 1
                                else "order from transitions"))
    if len(anchored) > 1:
        return anchored[0], ("AMBIGUOUS — multiple head-anchored orders fit")
    return good[0], "AMBIGUOUS — no order anchored on the window head"


def decode_mr4(mr4):
    """JESD235D Table 15: MR4 = OP7/OP6 reserved, OP5 ERL, OP4 EWL,
    OP[3:2] Parity Latency (0-3 nCK), OP[1:0] DM/ECC."""
    op = mr4 & 3
    mode = {0: "DM", 2: "NONE", 3: "ECC"}.get(op, "RESERVED")
    expl = {0: "DM on, ECC off (check bits not driven back)",
            2: "DM off, ECC off (no mask, no check bits)",
            3: "DM off, ECC on (DM-pin cells carry host-computed check bits)"
            }.get(op, "OP=01 is reserved (DM and ECC cannot coexist)")
    return (f"MR4=0x{mr4:02x}: ERL={mr4 >> 5 & 1} EWL={mr4 >> 4 & 1} "
            f"PL={(mr4 >> 2) & 3} OP[1:0]={op:02b}  ->  mode = {mode}\n"
            f"        ({expl})")


def query_hbm_mode(card, args):
    """Arm the read-only 10h WDR per FBPA until a clean 3-chunk frame.

    Returns (frame96, fbpa, attempts, note) or (None, reason, ...).
    The frame is one global stream tapped per FBPA (§11.8/§11.9), so the
    first clean capture answers for the card.
    """
    live = live_fbpas(card.rd)
    if not live:
        return None, "no live FBPA (cannot query)", None
    sensors0 = hbm_sentinels(card.rd)
    fbpa_cap = max(1, min(args.fbpa_max, len(live)))
    last_note = ""
    saw_capture = False   # at least one arm actually ran (INSTR landed)
    for f in live[:fbpa_cap]:
        for a in range(1, args.attempts + 1):
            words, status, note = wdr_arm_capture(
                card.rd, card.wr, f, 0x10, 5, True,
                mode_arm=WDR_MODE_STREAM, channel=0)
            last_note = f"fbpa={f} attempt={a}: {status} {note}"
            if args.trace:
                for k, (d, si, sd, st) in enumerate(words or []):
                    print(f"        [{k}] DATA=0x{d:08x} SHWIR=0x{si:08x} "
                          f"SHWDR=0x{sd:08x} STAT=0x{st:08x}")
            if words is None:
                # DROPPED can be a sub-ms ISR re-arm reading back the
                # ISR's own value — only "gate closed" after EVERY FBPA
                # in the budget dropped every attempt.
                if status != "DROPPED":
                    saw_capture = True
                time.sleep(0.05)
                continue
            saw_capture = True
            cycle, how = mrdump_align([d for d, _, _, _ in words])
            if cycle is not None and len(cycle) == 3 and \
                    "AMBIGUOUS" not in how.upper():
                frame = 0
                for p, c in enumerate(cycle):
                    frame |= c << (32 * p)
                new = hbm_sentinels(card.rd)
                if new != sensors0:
                    return None, (f"ABORT: sentinels {sensors0} -> {new} "
                                  f"— reboot the GPU", last_note)
                dx = xid_since(args.xid0)
                if dx:
                    return None, f"ABORT: new Xid: {dx} — reboot the GPU", \
                        last_note
                return frame, (f, a, how), None
            time.sleep(0.05)
    if not saw_capture:
        return None, ("I1500 write gate closed (every INSTR write dropped "
                      "— SEC2 booter PLM opens not installed); HBM mode "
                      "not readable", last_note)
    return None, f"no clean 3-chunk frame in budget ({last_note})", None


# ---------------------------------------------------------------- sections
def present_str(robit, ro, enabled, known):
    if robit is None:
        if enabled:
            return "YES (no robit; alive while enabled)"
        return "yes (known 2026-09-06)" if known == "present" \
            else "ABSENT (known 2026-09-06)"
    lit = (ro >> robit) & 1
    if enabled:
        return (f"YES (robit {robit} lit)" if lit
                else f"NO — robit {robit} dark while enabled (check boot)")
    if lit:
        return f"YES — robit {robit} lit while disabled (anomaly)"
    return "yes (known 2026-09-06)" if known == "present" \
        else "ABSENT (2026-09-06 bisection)"


def probe_sram(card, ro, v0, v1, v2):
    """Print the SRAM unit table; return (n_enabled, units) list."""
    print(f"\n-- SRAM ECC units -------------------------------------------------")
    print(f"{'unit':22} {'reg':10} ovr val  enabled  present")
    rows = []
    for i, name, robit, known in ECC0:
        if i == 4:          # DRAM handled in the DRAM section
            continue
        val = (v0 >> (4 * i)) & 0x7
        ovr = (v0 >> (4 * i + 3)) & 1
        rows.append((name, val, ovr, known))
    for db, name, robit, known in ECC1:
        val = (v1 >> db) & 1
        ovr = (v1 >> (db + 1)) & 1
        rows.append((name, val, ovr, known))
    for db, name, known in ECC2:
        val = (v2 >> db) & 1
        ovr = (v2 >> (db + 1)) & 1
        rows.append((name, val, ovr, known))

    robit_of = {n: r for _, n, r, _ in ECC0}
    robit_of.update({n: r for _, n, r, _ in ECC1})
    robit_of.update({n: None for _, n, _ in ECC2})
    reg_of = {}
    for i, name, _, _ in ECC0:
        if i != 4:
            reg_of[name] = "0x82380c"
    for db, name, _, _ in ECC1:
        reg_of[name] = "0x823810"
    for db, name, _ in ECC2:
        reg_of[name] = "0x82382c"

    n_en = 0
    for name, val, ovr, known in rows:
        enabled = val != 0
        n_en += enabled
        print(f"{name:22} {reg_of[name]:10} {ovr:<4} {val:<4}  "
              f"{'YES' if enabled else 'no':<7}  "
              f"{present_str(robit_of[name], ro, enabled, known)}")
    present = [n for n, v, o, k in rows if k == "present"]
    absent = [n for n, v, o, k in rows if k == "ABSENT"]
    print(f"SRAM: {n_en}/{len(rows)} units enabled | "
          f"present {len(present)}, absent {len(absent)} "
          f"({', '.join(absent) if absent else 'none'})")
    if (v0, v1, v2) == (STOCK[R_FEAT_ECC0], STOCK[R_FEAT_ECC1],
                        STOCK[R_FEAT_ECC2]):
        print("(override regs at stock — no SRAM ECC enabled this boot)")
    return n_en, rows


def probe_dram(card, ro, v0):
    """Print the DRAM ECC section; return 'ENABLED+ARMED'/'ENABLED'/'off'."""
    i = 4
    val = (v0 >> (4 * i)) & 0x7
    ovr = (v0 >> (4 * i + 3)) & 1
    enabled = val != 0
    lit = (ro >> 16) & 1
    fuse = card.rd(R_FUSE_ECC)
    fbpactrl = card.rd(R_FBPA_CTRL)

    print(f"\n-- DRAM ECC --------------------------------------------------------")
    print(f"{'unit':22} {'reg':10} ovr val  enabled  present")
    print(f"{'DRAM':22} {'0x82380c':10} {ovr:<4} {val:<4}  "
          f"{'YES' if enabled else 'no':<7}  "
          f"{'YES (robit 16 lit)' if lit else 'no (robit 16 dark)'}")
    print(f"FUSE_ECC_EN 0x820228     = 0x{fuse:08x}"
          f"   (OTP: {'on' if fuse else 'OFF — datapath fused off, '
          'overridden by FEATURE_OVERRIDE_ECC'})")
    print(f"FBPA_ECC_CTRL 0x9A0470   = 0x{fbpactrl:08x} (broadcast)")

    live = live_fbpas(card.rd)
    bad = []
    for pt in live:
        for off, bits in zip(ARM_OFFS, ARM_BITS):
            if card.rd(FBPA_UC_BASE + off + pt * FBPA_UC_STRIDE) & bits \
                    != bits:
                bad.append((pt, off, bits))
                break
    g0 = card.rd(ARM_GLOBALS[0]) & ARM_GLOBAL_BIT
    g1 = card.rd(ARM_GLOBALS[1]) & ARM_GLOBAL_BIT
    print(f"sideband datapath arm (pre-GSP Booter RMW):")
    print(f"  partitions: {len(live) - len(bad)}/{len(live)} live armed "
          f"(+0x470&0x41 +0x1004&0x200 +0x33c&0x3 +0x84&0x1)")
    if bad:
        for pt, off, bits in bad:
            print(f"    ! pt{pt:2d} +0x{off:04x} mask 0x{bits:x} NOT set")
    print(f"  globals: 0x17E318&0x1 {'set' if g0 else 'NOT set'}, "
          f"0x17E21C&0x1 {'set' if g1 else 'NOT set'}")
    armed = not bad and g0 and g1
    if enabled and armed:
        print(f"DRAM ECC: ENABLED + ARMED"
              f"{' + readout bit 16' if lit else ''}")
        return "ENABLED+ARMED"
    if enabled:
        print(f"DRAM ECC: ENABLED but sideband arm INCOMPLETE")
        return "ENABLED"
    print(f"DRAM ECC: OFF (0x82380c DRAM nibble val=0)")
    return "off"


def probe_hbm(card, args):
    """Print the HBM die-mode section; return 'ECC'/'DM'/'NONE'/... or
    'not queried' / 'unavailable'."""
    print(f"\n-- HBM die mode (I1500 MODE_REGISTER_DUMP, WIR 0x0010 ch0) --")
    if args.read_only:
        print("  not queried (--read-only: no I1500 arm issued)")
        return "not queried"
    t0 = time.monotonic()
    frame, meta, err = query_hbm_mode(card, args)
    dt = time.monotonic() - t0
    if err:
        print(f"  {err}  [{dt:.1f}s]")
        if meta:
            print(f"  last: {meta}")
        return "unavailable"
    f, a, how = meta
    mrs = [(frame >> (8 * k)) & 0xFF for k in range(12)]
    print(f"  capture ok: fbpa={f} attempt={a} ({how})  [{dt:.1f}s]")
    print(f"  96-bit MR image = 0x{frame:024x}")
    for k in range(0, 12, 4):
        print("  " + "  ".join(f"MR{m:<2}=0x{mrs[m]:02x}" for m in range(k, k + 4)))
    print("  " + decode_mr4(mrs[4]))
    return {0: "DM", 2: "NONE", 3: "ECC"}.get(mrs[4] & 3, "RESERVED")


def probe_card(card, args):
    ro = card.rd(R_FEAT_RO)
    v0 = card.rd(R_FEAT_ECC0)
    v1 = card.rd(R_FEAT_ECC1)
    v2 = card.rd(R_FEAT_ECC2)
    print()
    print("=" * 72)
    print(f"card 0000:{card.name}  (gpcprobe idx {card.idx})")
    print("=" * 72)
    print(f"FEATURE_READOUT 0x823814 = 0x{ro:08x}")
    print(f"FEATURE_OVERRIDE_ECC    0x82380c = 0x{v0:08x}")
    print(f"FEATURE_OVERRIDE_ECC_1  0x823810 = 0x{v1:08x}")
    print(f"FEATURE_OVERRIDE_ECC_2  0x82382c = 0x{v2:08x}")
    n_en, _ = probe_sram(card, ro, v0, v1, v2)
    dram = probe_dram(card, ro, v0)
    hbm = probe_hbm(card, args)
    print()
    print(f"RESULT: SRAM ECC {'on' if n_en else 'off'} "
          f"({n_en} units) | DRAM ECC {dram} | HBM die mode {hbm}")
    return hbm


def main():
    ap = argparse.ArgumentParser(
        description="CMP 170HX ECC unit probe (BAR0 + I1500, no nvidia-smi)")
    ap.add_argument("--card", type=int, action="append", default=None,
                    help="gpcprobe card index (default: all)")
    ap.add_argument("--read-only", action="store_true",
                    help="no I1500 arm: HBM die mode not queried, zero writes")
    ap.add_argument("--attempts", type=int, default=4,
                    help="clean-capture attempts per FBPA (default 4)")
    ap.add_argument("--fbpa-max", type=int, default=3,
                    help="max live FBPAs to try for the HBM query (default 3)")
    ap.add_argument("--trace", action="store_true",
                    help="print raw DATA/SHWIR/SHWDR/STAT words per attempt")
    args = ap.parse_args()

    if os.geteuid() != 0:
        sys.exit(f"need root (the {DEV} node is 0600)")
    try:
        fd = os.open(DEV, os.O_RDWR)
    except OSError as e:
        sys.exit(f"cannot open {DEV}: {e} — is gpcprobe.ko loaded? "
                 f"(make && insmod in {os.path.dirname(os.path.abspath(__file__))})")

    info = bytearray(4 + 16 * 4)
    fcntl.ioctl(fd, GP_IOC_INFO, info)
    count = struct.unpack_from("<I", info, 0)[0]
    bdfs = [struct.unpack_from("<I", info, 4 + 4 * i)[0] for i in range(count)]
    if not count:
        sys.exit("gpcprobe found no NVIDIA GPU BAR0")

    args.xid0 = None
    try:
        import subprocess
        args.xid0 = subprocess.run(["dmesg"], capture_output=True,
                                   text=True, timeout=10).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        args.xid0 = []

    print(f"eccprobe — CMP 170HX (GA100) ECC state")
    print(f"source: BAR0 via {DEV} + I1500 bridge — no nvidia-smi")
    print(f"cards: " + ", ".join(f"idx {i} = 0000:{(b >> 8) & 0xFF:02x}:"
                                 f"{(b >> 3) & 0x1F:02x}.{b & 7}"
                                 for i, b in enumerate(bdfs)))
    for idx, bdf in enumerate(bdfs):
        if args.card is not None and idx not in args.card:
            continue
        probe_card(Card(fd, idx, bdf), args)
    print()


if __name__ == "__main__":
    main()
