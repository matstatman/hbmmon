"""gpc_probe — /dev/gpcprobe ioctl constants + the I1500 WDR capture
engine for hbmmon (CMP 170HX, GA100/A100).

No CLI of its own — hbmmon.py is the tool. It imports this module for
the ioctl numbers, the HBM sentinel/Xid abort criteria and
wdr_arm_capture: the signed-off §11.8 on-demand WDR request recipe
(170hx repo docs/hbm-1500-170hx.md §11.7/§11.8, live-verified
2026-09-19, sweep signed off 2026-09-20).

On-demand WDR request engine: the SEC2 GSP booter opens the nine
I1500-neighborhood PLMs at L3 on every boot, so host (PL0) writes to
INSTR/MODE land at runtime and WIRs are accepted and executed by the FB
Falcon. Multi-word WDRs walk the 32-bit DATA latch LSB-first, one
32-bit chunk per DATA read (read-to-advance, §11.3); the 82-bit
DEVICE_ID's final chunk is clamped (overlap law Z[13:0] == Y[31:18]).

Only READ-type 1500 instructions are ever armed through this engine
(0E DEVICE_ID / 0F TEMPERATURE / 10 MR-dump) — never reset/repair/MBIST.
The kernel module allowlists the INSTR/MODE control registers as the
only writable BAR0 offsets; the DATA latch is read-only from the host.
"""

import itertools
import subprocess
import time

GP_IOC_MAGIC = 0x47  # 'G'
GP_IOC_INFO = (0x00 << 30) | (0 << 16) | (GP_IOC_MAGIC << 8) | 1   # _IO('G', 1)
GP_IOC_READ32 = (3 << 30) | (12 << 16) | (GP_IOC_MAGIC << 8) | 2   # _IOWR('G', 2, 12B)
# _IOW: dir bits = 1 (arg written into kernel); 2 = read out; 3 = both
GP_IOC_WRITE = (1 << 30) | (12 << 16) | (GP_IOC_MAGIC << 8) | 3    # _IOW('G', 3, 12B)

# HBM sentinels diffed before/after every armed FBPA of the sweep — a
# change means the write reached something it should not (170hx repo
# docs/hbm-1500-170hx.md §11.8).
SENTINELS = {
    "FBPA_NUM_ACTIVE": 0x009A0164,
    "FBPA_TRAINING":   0x009A0974,
}

# Per-FBPA unicast I1500 aperture (170hx repo docs/hbm-1500-170hx.md
# §11.5). Same in-aperture layout as the 0x009A0000 broadcast,
# replicated 24 times; a floorswept FBPA's aperture reads 0xBADFxxxx,
# so the live map is self-describing.
FBPA_UC_BASE   = 0x00900000
FBPA_UC_STRIDE = 0x00004000
FBPA_UC_COUNT  = 24
FBPA_UC_CFG1   = 0x0204          # aperture-relative
FBPA_UC_DATA   = 0x3CBC          # aperture-relative (= I1500 DATA)

# On-demand WDR request engine (2026-09-19, §11.7). Channel 0 is the
# Falcon's runtime poll target; 0x0F (all channels) is the only channel
# verified to execute read-type WDRs (boot used 0x0F0E).
WDR_CH0 = 0x00                    # channel 0 (the Falcon's runtime poll target)
WDR_WDR_CH = 0x0F                 # all-channels — the only channel verified to
                                  # execute read-type WDRs (boot used 0x0F0E)
WDR_MODE_ARM = 0x0000008          # the Falcon's runtime temp-poll mode: the
                                  # bridge executes ONLY the 0F WDR in this mode
                                  # (proven 2026-09-19: WIR 0x000E sat in the
                                  # DRAM 44 ms, SHWDR still temp-format parity)
WDR_MODE_STREAM = 0x00000052      # the Falcon's boot/stream mode — required for
                                  # 0E/10 to execute (C1 replicated the boot
                                  # regime bit-exact, 2026-09-19)
WDR_SETTLE_S = 0.005              # transfer completes <1 ms after the INSTR
                                  # write; must stay under the ISR re-arm
                                  # (~24 ms stream / ~45 ms poll, 2026-09-19)
WDR_STEP_S = 0.01                 # between frame-word reads (tight-loop safe)
WDR_UC_INSTR = 0x3CB4             # aperture-relative I1500 control offsets
WDR_UC_MODE = 0x3CB8
WDR_UC_DATA = 0x3CBC
WDR_UC_SHWIR = 0x3CC0
WDR_UC_SHWDR = 0x3CC4
WDR_UC_STAT = 0x3CC8


def xid_since(dmesg_before):
    """Return new 'Xid' lines from dmesg since the baseline (or [])."""
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True,
                             timeout=10).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return ["<dmesg unavailable>"]
    new = [l for l in out if "Xid" in l and l not in dmesg_before]
    return new


def hbm_sentinels(rd):
    return {name: rd(off) for name, off in SENTINELS.items()}


def _wdr_frame_order(words3):
    """Distinct (X, Y, Z) orderings of 3 frame words satisfying the
    clamped-overlap law Z[13:0] == Y[31:18] (§11.3). Inlined from
    decode_deviceid.solve_frame to keep this module dependency-free.
    The law is 14 bits, so a wrong ordering passing it is ~1-in-16k."""
    out = set()
    for perm in itertools.permutations(range(3)):
        X, Y, Z = (words3[i][0] for i in perm)
        if (Z & 0x3FFF) == (Y >> 18):
            out.add((X, Y, Z))
    return sorted(out)


def wdr_arm_capture(rd, wr, fbpa, opcode, n_read, write_ok,
                    mode_arm=WDR_MODE_ARM, channel=WDR_CH0,
                    max_retries=2):
    """Arm one READ-only WDR on a live FBPA's unicast I1500 instance.

    Protocol (boot-regime replication, live-verified 2026-09-19 §11.8):
    save INSTR/MODE, write MODE = arm value (stream mode 0x52 — the bridge
    will NOT execute 0E/10 while MODE holds the 0x08 temp-poll value),
    then INSTR = WIR (channel + opcode; the WIR shifts into the DRAM
    within ~1 ms), settle (must beat the Falcon ISR's ~24 ms re-arm),
    read n_read x (DATA, SHWIR, SHWDR, STAT) — read-to-advance walks
    multi-word frames (observed order X, Z, Y) — restore INSTR then MODE,
    verify, report. Returns (words, status, note); words is None unless
    the capture actually ran. Only the WDR-safe read opcodes are ever
    armed (0E/0F/10) — never reset/repair/MBIST.

    2026-09-19 trace facts baked into this design: the unicast INSTR
    register IS the global WIR master (the broadcast readout follows a
    unicast write); per-FBPA DATA latches are per-stack taps of the one
    stream; MODE is re-armed by the host only (the ISR touches INSTR).

    ISR re-arm race (measured 2026-09-20 sweep, 9/32 FBPAs): the Falcon's
    poller can replace the armed WIR with its own 0x0F any time within
    the ~24 ms period — including before the first read. A capture is
    trusted only while every SHWIR reading holds the armed echo
    (WIR << 4); otherwise the WDR is re-armed (INSTR write only — MODE
    is untouched by the ISR) and the read budget is retaken, up to
    max_retries times. DROPPED/PARTIAL readbacks are retried for the
    same reason: a re-arm inside the sub-ms arm/readback window reads
    back the ISR's own value.
    """
    base = FBPA_UC_BASE + fbpa * FBPA_UC_STRIDE
    IN, MO = base + WDR_UC_INSTR, base + WDR_UC_MODE
    DA, SHI, SHD, ST = (base + o for o in
                        (WDR_UC_DATA, WDR_UC_SHWIR, WDR_UC_SHWDR, WDR_UC_STAT))
    wir = (channel << 8) | opcode
    echo = wir << 4
    i0, m0 = rd(IN), rd(MO)
    mode_write = m0 != mode_arm
    if not write_ok:
        return None, "dry-run", (
            f"saved INSTR=0x{i0:08x} MODE=0x{m0:08x}; plan"
            + (f" MODE<-0x{mode_arm:08x}," if mode_write else "")
            + f" INSTR<-0x{wir:08x}; capture {n_read} words"
            + (f" (ISR-race re-arm x{max_retries} max);" if max_retries else "")
            + " restore+verify; sentinels/Xid")
    if mode_write:
        wr(MO, mode_arm)
    words, status, note = None, "DROPPED", ""
    for attempt in range(1 + max_retries):
        wr(IN, wir)
        i1, shi1 = rd(IN), rd(SHI)
        if i1 == i0 and i0 != wir:
            words, status, note = None, "DROPPED", (
                f"INSTR readback 0x{i1:08x} = saved (ISR re-arm before "
                f"readback, or gate closed?) on attempt {attempt + 1}")
            continue
        if i1 != wir:
            words, status, note = None, "PARTIAL", (
                f"INSTR readback 0x{i1:08x} (neither saved nor armed) "
                f"on attempt {attempt + 1}")
            continue
        time.sleep(WDR_SETTLE_S)
        words = []
        for _ in range(n_read):
            words.append((rd(DA), rd(SHI), rd(SHD), rd(ST)))
            time.sleep(WDR_STEP_S)
        if all(si == echo for _, si, _, _ in words):
            status = "OK"
            note = (f"SHWIR echo 0x{echo:05x} held all {n_read} reads"
                    + (f" after {attempt} ISR re-arm(s)" if attempt else ""))
            break
        status = "ISR-RACE"
        note = (f"Falcon ISR re-armed the WIR mid-capture (SHWIR left "
                f"0x{echo:05x}) on attempt {attempt + 1}; re-arming")
    wr(IN, i0)
    if mode_write:
        wr(MO, m0)
    time.sleep(0.05)
    fin_in, fin_mo = rd(IN), rd(MO)
    if (fin_in, fin_mo) != (i0, m0):
        status = (f"RESTORE-MISMATCH final INSTR=0x{fin_in:08x} "
                  f"MODE=0x{fin_mo:08x} (Falcon self-heal or stuck)")
    return words, status, note
