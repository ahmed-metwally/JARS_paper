# -*- coding: utf-8 -*-
"""Create ONE shared initialisation for everything outside the ViT, so that
arms B and C start their fusion blocks, skip projections, decoder and head
from identical weights.

Why this exists, as measured rather than assumed: building the encoder by
from_pretrained() and by Dinov2Model(config) leaves torch's CPU generator in
different states (numpy and python.random are unaffected). Every module
constructed after the encoder therefore draws different numbers -- 73.61M
parameters across 152 tensors. Without this control, a B-vs-C difference
would mix the transferred representation with ordinary initialisation
variance.

The source draw is arbitrary but FIXED and recorded: the model is built once
under arm C's configuration at seed 42, and its non-ViT tensors are saved.
Which draw it is does not matter; that both arms use the same one does.

Writes shared_init_nonencoder.pt and prints its SHA-256.
NOTHING TRAINS HERE.
"""
import io, os, sys, json, hashlib, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "shared_init_nonencoder.pt")
SRC_ARM, SRC_SEED = "C_pretrained_frozen", 42

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
sys.path.insert(0, ROOT)
_rt = os.path.join(ROOT, "run_runtime_preflight.py")

# reuse the preflight's own prefix loader so the model is built exactly the
# way the run builds it
import json as _j, re as _re, tempfile


def _is_top_def(l):
    return bool(l) and not l[0].isspace() and (l.startswith("class ") or l.startswith("def "))


def _prefix(key, seed=42):
    p = os.path.join(ROOT, "runs", "CLCD_%s_seed_%d.ipynb" % (key, seed))
    L = "".join("".join(c["source"]) for c in
                _j.load(io.open(p, encoding="utf-8"))["cells"]
                if c["cell_type"] == "code").split("\n")
    j = next(k for k, l in enumerate(L) if l.startswith("def main("))
    head = "\n".join(L[:j])
    assert "def main(" not in head and "if __name__" not in head
    tmp = tempfile.mkdtemp(prefix="mkinit_")
    head, n = _re.subn(r'(?m)^OUTPUT_BASE = .*$',
                       'OUTPUT_BASE = r"%s"' % tmp.replace("\\", "\\\\"), head, count=1)
    assert n == 1
    return head, p


if __name__ == "__main__":
    if os.path.exists(OUT):
        h = hashlib.sha256(io.open(OUT, "rb").read()).hexdigest()
        print("REFUSING to overwrite an existing shared initialisation.")
        print("  %s" % OUT)
        print("  sha256 = %s" % h)
        print("  Delete it by hand if you really mean to replace it; every arm")
        print("  already trained against it would otherwise become incomparable.")
        sys.exit(1)

    head, src_path = _prefix(SRC_ARM, SRC_SEED)
    ns = {"__name__": "mkinit"}
    exec(compile(head, src_path, "exec"), ns)
    ns["set_seed"](ns["SEED"])
    model = ns["SiameseViT"]()

    t = dict(model.named_parameters())
    t.update(dict(model.named_buffers()))
    keep = {k: v.detach().cpu().clone() for k, v in t.items() if ".vit." not in k}
    drop = [k for k in t if ".vit." in k]
    nparam = sum(v.numel() for v in keep.values())

    torch.save({"tensors": keep,
                "meta": {"source_arm": SRC_ARM, "source_seed": SRC_SEED,
                         "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                         "torch": torch.__version__,
                         "n_tensors": len(keep), "n_params": int(nparam),
                         "excluded_vit_tensors": len(drop)}}, OUT)

    h = hashlib.sha256(io.open(OUT, "rb").read()).hexdigest()
    io.open(os.path.join(ROOT, "shared_init_manifest.json"), "w", encoding="utf-8").write(
        json.dumps({"file": "shared_init_nonencoder.pt", "sha256": h,
                    "sha256_16": h[:16], "n_tensors": len(keep),
                    "n_params": int(nparam), "excluded_vit_tensors": len(drop),
                    "source_arm": SRC_ARM, "source_seed": SRC_SEED,
                    "torch": torch.__version__,
                    "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")},
                   indent=1))
    print("wrote %s" % OUT)
    print("  tensors kept (outside ViT) : %d  (%.2fM params)" % (len(keep), nparam / 1e6))
    print("  ViT tensors excluded       : %d" % len(drop))
    print("  sha256                     : %s" % h)
    print("  sha256[:16]                : %s" % h[:16])
    print("\nNothing trained. Next: regenerate the notebooks, then verify.")
