# -*- coding: utf-8 -*-
"""Acceptance pipeline for one finished arm. CPU only -- safe to run while
another arm is training on the GPU.

    python accept_C.py                      # defaults to arm C, seed 42
    python accept_C.py CLCD_B_random_frozen_seed_42

Writes everything to accept_<run>.txt as well as the screen, so the result is
on disk even if the console scrolls or the window is closed.
"""
import os, sys, io, datetime

# hard guarantee this process never touches the GPU the training job holds
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" is treated as unset on Windows

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RUN = sys.argv[1] if len(sys.argv) > 1 else "CLCD_C_pretrained_frozen_seed_42"
OUT = os.path.join(ROOT, "accept_%s.txt" % RUN)


class Tee:
    def __init__(self, path):
        self.f = io.open(path, "w", encoding="utf-8")
        self.t = sys.stdout
    def write(self, s):
        self.t.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.t.flush(); self.f.flush()


sys.stdout = sys.stderr = Tee(OUT)
print("acceptance pipeline  |  run = %s" % RUN)
print("started %s UTC  |  CUDA_VISIBLE_DEVICES=%r"
      % (datetime.datetime.utcnow().isoformat(timespec="seconds"),
         os.environ["CUDA_VISIBLE_DEVICES"]))
print("=" * 78)

import preflight_clcd as pf

for step, fn, args in (
        ("1/4  post_run_weight_integrity", pf.post_run_weight_integrity, (RUN,)),
        ("2/4  prediction_audit",          pf.prediction_audit,          ()),
        ("3/4  archive_evidence",          pf.archive_evidence,          (RUN,)),
        ("4/4  accept_run",                pf.accept_run,                (RUN,)),
):
    print("\n" + "-" * 78)
    print(step)
    print("-" * 78)
    try:
        fn(*args)
    except Exception as e:
        import traceback
        print("STEP RAISED: %r" % (e,))
        traceback.print_exc()
        print("\nstopping here -- fix this before running the next arm.")
        break

print("\n" + "=" * 78)
print("written to %s" % OUT)
