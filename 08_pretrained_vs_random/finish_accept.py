# -*- coding: utf-8 -*-
"""Finish the acceptance pipeline for one arm -- one step per PROCESS.

    python finish_accept.py integrity CLCD_C_pretrained_frozen_seed_42
    python finish_accept.py freeze    CLCD_B_random_frozen_seed_42
    python finish_accept.py finish    <RUN>

Three reasons this is a script and not notebook cells.

1. post_run_weight_integrity loads the 1.3 GB checkpoint and the released
   DINOv2 reference at the same time, and accept_run calls it a SECOND time
   for any arm whose name contains "frozen". Two such peaks in one kernel and
   the OS kills it, so Jupyter shows only "Kernel Dead" with no traceback.
   Here each step is its own process and the verdict travels on disk.

2. stderr is captured into the transcript as well as stdout. The first
   version of this script tee'd stdout only, so when the arm-B step died its
   traceback went nowhere and the transcript ended after the header.

3. "integrity" is the WRONG CHECK FOR A RANDOMLY INITIALIZED ARM.
   post_run_weight_integrity compares the saved encoder against the released
   facebook/dinov2-base weights. That is the right question for arm C
   (ENCODER_INIT="pretrained", frozen): did the frozen pretrained encoder
   stay equal to what was loaded. Arm B is ENCODER_INIT="random", frozen --
   its encoder was never DINOv2, so that comparison reports every tensor as
   "changed" and prints "the freeze did NOT hold", which is false. The freeze
   question for arm B is whether the encoder moved DURING TRAINING, and the
   evidence for that is the epoch-zero checkpoint against the final one.
   That is what "freeze" measures, and it needs no reference download.

Both steps write the same verdict file, with the method recorded in it, so
the acceptance gate states which proof it was given.
"""
import os, sys, io, json, gc, hashlib, datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # must precede "import torch"

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STEPS = ("integrity", "freeze", "finish")
if len(sys.argv) != 3 or sys.argv[1] not in STEPS:
    print(__doc__)
    raise SystemExit(2)
STEP, RUN = sys.argv[1], sys.argv[2]

VERDICT = os.path.join(ROOT, "integrity_%s.json" % RUN)
LOG     = os.path.join(ROOT, "accept_%s.txt" % RUN)
FINAL   = os.path.join(ROOT, "runs_out", RUN, "final")

_f = io.open(LOG, "a", encoding="utf-8")


class _Tee(object):
    """stdout AND stderr -> console + transcript. A traceback is evidence."""

    def __init__(self, stream):
        self._s = stream

    def write(self, s):
        self._s.write(s)
        _f.write(s)
        _f.flush()

    def flush(self):
        self._s.flush()
        _f.flush()


sys.stdout = _Tee(sys.__stdout__)
sys.stderr = _Tee(sys.__stderr__)

import torch
torch.set_grad_enabled(False)
import preflight_clcd as pf

print("\n" + "#" * 78)
print("# %-10s run: %s" % (STEP, RUN))
print("# %s UTC | python %s | torch %s | GPU hidden: %s | grad off: %s"
      % (datetime.datetime.utcnow().isoformat(timespec="seconds"),
         sys.version.split()[0], torch.__version__,
         not torch.cuda.is_available(), not torch.is_grad_enabled()))
print("#" * 78)

PRE = "encoder.vit."


def _record(unchanged, method, detail):
    io.open(VERDICT, "w", encoding="utf-8").write(json.dumps(
        {"run_name": RUN,
         "encoder_unchanged": bool(unchanged),
         "method": method,
         "detail": detail,
         "measured_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
         "measured_by": "finish_accept.py " + STEP,
         "python": sys.version.split()[0],
         "torch": torch.__version__}, indent=1, ensure_ascii=False))
    print("\nverdict recorded -> %s" % VERDICT)


def _hashes(path):
    """SHA-256 per encoder tensor, one checkpoint in memory at a time."""
    import numpy as np
    sd = torch.load(path, map_location="cpu", weights_only=False)
    n_opt = len(sd.get("optimizer_state_dict", {}).get("state", {})) \
        if isinstance(sd, dict) else 0
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    enc, other, other_params = {}, 0, 0
    for k, v in sd.items():
        if not hasattr(v, "detach"):
            continue
        if k.startswith(PRE):
            a = np.ascontiguousarray(v.detach().cpu().float().numpy())
            enc[k[len(PRE):]] = hashlib.sha256(a.tobytes()).hexdigest()
            del a
        else:
            other += 1
            other_params += v.numel()
    del sd
    gc.collect()
    return enc, n_opt, other, other_params


# ────────────────── step: encoder vs released DINOv2 (arm C) ──────────────────
if STEP == "integrity":
    ok = pf.post_run_weight_integrity(RUN)
    if ok is None:
        print("\nno checkpoint found -- nothing recorded")
        raise SystemExit(1)
    _record(ok, "encoder vs released facebook/dinov2-base",
            "every encoder tensor in the final checkpoint compared against "
            "the released weights")
    raise SystemExit(0 if ok else 1)

# ────────────── step: epoch-zero vs final checkpoint (arm B) ──────────────
if STEP == "freeze":
    a = os.path.join(FINAL, "ckpt_epoch_000.pth")
    b = os.path.join(FINAL, "best_model.pth")
    for q in (a, b):
        if not os.path.exists(q):
            print("missing checkpoint: %s" % q)
            raise SystemExit(1)
    print("epoch-zero : %s  (%d bytes)" % (os.path.basename(a), os.path.getsize(a)))
    print("final      : %s  (%d bytes)" % (os.path.basename(b), os.path.getsize(b)))
    print("\nhashing epoch-zero encoder tensors ...")
    h0, opt0, oth0, par0 = _hashes(a)
    print("hashing final encoder tensors ...")
    h1, opt1, oth1, par1 = _hashes(b)

    common = sorted(set(h0) & set(h1))
    moved = [k for k in common if h0[k] != h1[k]]
    print("\nDURING training -- encoder at epoch 0 vs encoder at the end")
    print("  encoder tensors, epoch 0   : %d" % len(h0))
    print("  encoder tensors, final     : %d" % len(h1))
    print("  tensors compared           : %d" % len(common))
    print("  tensors byte-identical     : %d" % (len(common) - len(moved)))
    print("  tensors changed            : %d" % len(moved))
    print("  optimizer state entries    : epoch 0 %d, final %d" % (opt0, opt1))
    print("  non-encoder tensors (final): %d  (%s parameters)"
          % (oth1, format(par1, ",")))
    if moved:
        print("  first changed tensors:")
        for k in moved[:5]:
            print("    %s" % k)
    only = sorted(set(h0) ^ set(h1))
    if only:
        print("  WARNING: %d tensor name(s) present on one side only" % len(only))
    ok = (not moved) and len(common) == len(h0) == len(h1) and not only
    print("=> %s" % ("UNCHANGED -- the encoder never moved during training"
                     if ok else "CHANGED -- the freeze did NOT hold"))
    _record(ok, "encoder at epoch 0 vs encoder at the end of training",
            "%d of %d encoder tensors byte-identical by SHA-256; "
            "no reference weights involved, which is why this is the correct "
            "proof for a randomly initialized frozen encoder"
            % (len(common) - len(moved), len(common)))
    raise SystemExit(0 if ok else 1)

# ───────────────── step: prediction audit, archive, gate ─────────────────
if "frozen" in RUN:
    if not os.path.exists(VERDICT):
        hint = "freeze" if "random" in RUN else "integrity"
        print("\nREFUSED: no recorded freeze verdict for %s.\n"
              "This arm holds its encoder frozen, so that check is not optional.\n"
              "Run it first, in its own process:\n\n"
              "    python finish_accept.py %s %s\n" % (RUN, hint, RUN))
        raise SystemExit(2)
    _rec = json.load(io.open(VERDICT, encoding="utf-8"))
    print("\nfreeze verdict: %s" % _rec["method"])
    print("  measured %s in a separate process -- encoder_unchanged = %s"
          % (_rec["measured_utc"], _rec["encoder_unchanged"]))

    def _measured(run_name=None, _v=_rec):
        """Give accept_run the verdict already measured, instead of letting it
        re-load the checkpoint and the reference in this process."""
        print("  (freeze: %s, measured %s in its own process)"
              % (_v["method"], _v["measured_utc"]))
        return _v["encoder_unchanged"]

    pf.post_run_weight_integrity = _measured

    # the gate's own detail string says "byte-identical to the released
    # weights", which is false for a randomly initialized arm. Give it the
    # method that was actually measured; the gate falls back to its own
    # wording when nothing sets this.
    _SHORT = {"encoder at epoch 0 vs encoder at the end of training":
              "unchanged, epoch 0 -> end of training",
              "encoder vs released facebook/dinov2-base":
              "byte-identical to released dinov2-base"}
    _m = _rec.get("method")
    if _m is None:   # verdict written before this field existed
        _m = ("encoder at epoch 0 vs encoder at the end of training"
              if "random" in RUN else "encoder vs released facebook/dinov2-base")
        print("  note: verdict file predates the 'method' field; "
              "method inferred from the arm as: %s" % _m)
    pf._FREEZE_DETAIL = _SHORT.get(_m, _m)[:46]

print()
pf.prediction_audit()
print()
pf.archive_evidence(RUN)
print()
accepted = pf.accept_run(RUN)
print("\ntranscript appended to %s" % LOG)
raise SystemExit(0 if accepted else 1)
