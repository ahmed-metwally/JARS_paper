# -*- coding: utf-8 -*-
# Build one notebook per symmetrization rule that measures the ROW SUMS of the
# cross-temporal attention maps on the full CLCD test split.
# Inference only.  No training.  Nothing already on disk is overwritten.
#
#   logit            S_12 = softmax(M),  S_21 = softmax(M^T)          -> rows sum to 1 by construction
#   attention_level  S_12 = (a12 + a21^T)/2,  S_21 = (a21 + a12^T)/2  -> row i sums to (1 + c_i)/2
#
# The mean row sum of the attention-level average is 1 identically (the column
# sums of a row-stochastic matrix add up to N), so the mean says nothing; what
# is measured is the deviation of the individual rows from 1.
import io, json, re, os

LROOT = os.path.join(os.path.expanduser("~"), "mnt", "Master Thesis Codes", "attention")
WROOT = r"D:\Master Thesis Codes\attention"
OUTDIR = os.path.join(LROOT, "ROWSUM")

def L(rel): return os.path.join(LROOT, *rel.replace("\\", "/").split("/"))
def W(rel): return WROOT + "\\" + rel

EVAL = r'''
# ===================================================================
#  ROW SUMS OF THE CROSS-TEMPORAL ATTENTION MAPS -- FULL CLCD TEST SPLIT
#  inference only, float32, model.eval()
#
#  A hook on every cross-attention module recomputes the two attention maps
#  from the module's own projections and the inputs it actually received,
#  and checks that they reproduce the module's real output before any
#  statistic is kept (parity).  For each map it records the sum of every row.
#
#  It also records how much the block's attention output would change if each
#  row were divided by its own sum: before the output LayerNorm that change is
#  |1 - 1/row_sum| by algebra; after the LayerNorm it is whatever the
#  LayerNorm leaves of it.
# ===================================================================
import os, csv, json, datetime, numpy as np, torch, torch.nn.functional as F

RS_VARIANT = __VAR__
CKPT       = __CKPT__
TEST_IMG1  = __TE1__
TEST_IMG2  = __TE2__
IMG_EXT    = __EXT__
OUT_CSV    = __OUT__
OUT_JSON   = __OUTJ__
MAX_PAIRS  = None          # None = full split; set to 4 for a smoke test
BATCH      = 1
TOLS       = (0.01, 0.05, 0.10)

assert os.path.exists(CKPT), "checkpoint not found: %s" % CKPT
_READ = globals().get("read_image_rasterio") or globals().get("read_image")
assert _READ is not None, "no image reader found in the source cell"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

f1s = load_files(TEST_IMG1, IMG_EXT)
f2s = load_files(TEST_IMG2, IMG_EXT)
assert len(f1s) == len(f2s) and len(f1s) > 0, (len(f1s), len(f2s))
if MAX_PAIRS: f1s, f2s = f1s[:MAX_PAIRS], f2s[:MAX_PAIRS]
print("rule: %s | test pairs: %d" % (RS_VARIANT, len(f1s)))

model = SiameseViT(dropout=0.26943406970601036).to(DEVICE)
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
sd = ck.get("model_state_dict", ck.get("state_dict", ck))
missing, unexpected = model.load_state_dict(sd, strict=False)
print("loaded | missing=%d unexpected=%d | saved best_f1=%s"
      % (len(missing), len(unexpected), ck.get("best_f1", "n/a")))
assert len(missing) == 0 and len(unexpected) == 0, "state_dict mismatch -- stop and report"
model.eval()
N_BLOCKS = len(model.cross_blocks)
assert N_BLOCKS > 0, "this variant has no cross-attention blocks"
print("cross-attention blocks: %d" % N_BLOCKS)

DIRS = ("S_12", "S_21")
ROWS = {(b, d): [] for b in range(N_BLOCKS) for d in DIRS}                 # row sums, one array per batch
REL  = {(b, d): np.zeros(3, dtype=np.float64) for b in range(N_BLOCKS) for d in DIRS}   # sum pre, sum post, n tokens
PARITY = [0.0]

def attention_maps(attn, x1, x2):
    """The two attention maps exactly as the module forms them, for the rule named in RS_VARIANT."""
    q1, k1, v1 = attn._split(attn.q1_proj(x1)), attn._split(attn.k1_proj(x1)), attn._split(attn.v1_proj(x1))
    q2, k2, v2 = attn._split(attn.q2_proj(x2)), attn._split(attn.k2_proj(x2)), attn._split(attn.v2_proj(x2))
    N = x1.shape[1]
    bias = attn.relative_bias_table[attn.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
    s12 = (q1 @ k2.transpose(-2, -1)) * attn.scale + bias
    s21 = (q2 @ k1.transpose(-2, -1)) * attn.scale + bias
    if RS_VARIANT == "logit":
        M = (s12 + s21.transpose(-2, -1)) / 2.0
        S12 = F.softmax(M, dim=-1)
        S21 = F.softmax(M.transpose(-2, -1), dim=-1)
    elif RS_VARIANT == "attention_level":
        a12 = F.softmax(s12, dim=-1)
        a21 = F.softmax(s21, dim=-1)
        S12 = 0.5 * (a12 + a21.transpose(-2, -1))
        S21 = 0.5 * (a21 + a12.transpose(-2, -1))
    else:
        raise ValueError(RS_VARIANT)
    return S12, S21, v1, v2

def make_hook(bi):
    def hook(attn, inp, out):
        x1, x2 = inp[0].float(), inp[1].float()
        S12, S21, v1, v2 = attention_maps(attn, x1, x2)
        for name, S, v, ref in (("S_12", S12, v2, out[0]), ("S_21", S21, v1, out[1])):
            rs = S.sum(dim=-1, dtype=torch.float64)                      # (B, heads, N)
            y  = attn._merge(S @ v)
            o  = attn.out_norm(attn.out_proj(y))                         # dropout is the identity in eval()
            PARITY[0] = max(PARITY[0], float((o - ref.float()).abs().max().item()))
            yn = attn._merge((S / rs.float().unsqueeze(-1)) @ v)         # every row divided by its own sum
            on = attn.out_norm(attn.out_proj(yn))
            pre  = (y - yn).norm(dim=-1) / (y.norm(dim=-1) + 1e-12)      # before out_proj and LayerNorm
            post = (o - on).norm(dim=-1) / (o.norm(dim=-1) + 1e-12)      # after the LayerNorm
            ROWS[(bi, name)].append(rs.float().cpu().numpy())
            REL[(bi, name)] += np.array([float(pre.sum().item()), float(post.sum().item()), float(pre.numel())])
        del S12, S21
    return hook

handles = [blk.attn.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.cross_blocks)]

from tqdm.auto import tqdm
with torch.no_grad():
    for s in tqdm(range(0, len(f1s), BATCH), desc="row sums"):
        b1 = [_READ(f1s[k]).transpose(2, 0, 1) for k in range(s, min(s + BATCH, len(f1s)))]
        b2 = [_READ(f2s[k]).transpose(2, 0, 1) for k in range(s, min(s + BATCH, len(f1s)))]
        x1 = torch.from_numpy(np.stack(b1)).float().to(DEVICE)
        x2 = torch.from_numpy(np.stack(b2)).float().to(DEVICE)
        _ = model(x1, x2)
        del x1, x2
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
for h in handles: h.remove()

assert PARITY[0] < 1e-3, ("the recomputed maps do not reproduce the module output: max|delta|=%.3e "
                          "-- stop, the harness is wrong, not the result" % PARITY[0])
print("\nparity  max|recomputed - module output| = %.3e   (must be ~0)" % PARITY[0])

def stats(arrs, rel=None):
    a = np.concatenate([x.reshape(-1) for x in arrs]).astype(np.float64)
    d = np.abs(a - 1.0)
    p = np.percentile(a, [1, 5, 50, 95, 99])
    out = dict(rows=int(a.size), mean=float(a.mean()), mean_abs_dev=float(d.mean()),
               max_abs_dev=float(d.max()), min=float(a.min()), max=float(a.max()),
               p01=float(p[0]), p05=float(p[1]), p50=float(p[2]), p95=float(p[3]), p99=float(p[4]))
    for t in TOLS: out["pct_within_%.2f" % t] = float(100.0 * np.mean(d <= t))
    if rel is not None:
        out["rel_change_before_layernorm"] = float(rel[0] / rel[2])
        out["rel_change_after_layernorm"]  = float(rel[1] / rel[2])
    return out

res = {}
for b in range(N_BLOCKS):
    for d in DIRS:
        res["block%d_%s" % (b + 1, d)] = stats(ROWS[(b, d)], REL[(b, d)])
allrel = sum(REL.values())
res["all"] = stats([x for v in ROWS.values() for x in v], allrel)

W1 = "%-14s %11s %10s %10s %8s %8s %8s %8s %9s %9s %10s %10s"
print("\n" + "=" * 132)
print(" ROW SUMS OF THE ATTENTION MAPS -- rule: %s -- %d pairs, float32" % (RS_VARIANT, len(f1s)))
print("=" * 132)
print(W1 % ("map", "mean|s-1|", "max|s-1|", "mean", "min", "p01", "p99", "max", "in+-0.01%", "in+-0.05%", "rel preLN", "rel postLN"))
print("-" * 132)
for k in list(res.keys()):
    m = res[k]
    print(W1 % (k, "%.6f" % m["mean_abs_dev"], "%.4f" % m["max_abs_dev"], "%.6f" % m["mean"], "%.4f" % m["min"],
                "%.4f" % m["p01"], "%.4f" % m["p99"], "%.4f" % m["max"], "%.2f" % m["pct_within_0.01"],
                "%.2f" % m["pct_within_0.05"], "%.5f" % m["rel_change_before_layernorm"], "%.5f" % m["rel_change_after_layernorm"]))
print("=" * 132)
print(" s = sum of one row of an attention map (one query token, one head).  %d rows in all." % res["all"]["rows"])
print(" rel preLN / postLN = mean relative change of the attention output if every row were divided by its sum,")
print(" before the output projection and after the LayerNorm that follows it.")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
cols = ["rows", "mean", "mean_abs_dev", "max_abs_dev", "min", "p01", "p05", "p50", "p95", "p99", "max"] + \
       ["pct_within_%.2f" % t for t in TOLS] + ["rel_change_before_layernorm", "rel_change_after_layernorm"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["rule", "map"] + cols)
    for k, m in res.items():
        w.writerow([RS_VARIANT, k] + ["%.8f" % m[c] if c != "rows" else m[c] for c in cols])
    w.writerow([RS_VARIANT, "_meta_pairs", len(f1s)])
    w.writerow([RS_VARIANT, "_meta_blocks", N_BLOCKS])
    w.writerow([RS_VARIANT, "_meta_parity", "%.3e" % PARITY[0]])
    w.writerow([RS_VARIANT, "_meta_ckpt", CKPT])
    w.writerow([RS_VARIANT, "_meta_written_utc", datetime.datetime.utcnow().isoformat(timespec="seconds")])
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({"rule": RS_VARIANT, "pairs": len(f1s), "blocks": N_BLOCKS, "precision": "float32",
               "parity_max_abs_delta": PARITY[0], "checkpoint": CKPT, "maps": res}, fh, indent=2)
print("written ->", OUT_CSV)
print("written ->", OUT_JSON)
'''

HOW = """# Row sums of the cross-temporal attention maps - {var}

Measures, on the trained checkpoint, how far the rows of the two attention
maps are from summing to one.

| rule | maps | row sum |
|---|---|---|
| logit (ours) | softmax(M), softmax(M^T) | 1 by construction |
| attention_level | (a12 + a21^T)/2, (a21 + a12^T)/2 | (1 + c_i)/2, c_i a column sum |

**Inference only. No training. Nothing is overwritten.** One forward pass per
pair over the 120 CLCD test pairs, float32.

Run the two cells in order. The second asserts that the recomputed maps
reproduce the module's real output before it reports anything; if that
assertion fires, stop - the harness is wrong, not the result.

> Do not run while a training job holds the GPU.
"""

RUNALL_MD = """# Run both row-sum notebooks

One cell. It executes `ROWSUM_CLCD_logit.ipynb` and
`ROWSUM_CLCD_attention_level.ipynb` in turn, in this same Python environment,
writes `rowsum_<rule>.csv` / `.json` beside them, and prints the two side by side.

Inference only. No training. Expect a few minutes per rule on the GPU.

If this cell cannot find `jupyter`, open each `ROWSUM_CLCD_*.ipynb` and press
Run All instead -- the result is identical.
"""

RUNALL = r'''
import os, sys, subprocess, time, json

FIXED = r"D:\Master Thesis Codes\attention\ROWSUM"
HERE  = os.getcwd()
CANDS = [FIXED, HERE, os.path.join(HERE, "ROWSUM")]
ROOT  = next((c for c in CANDS if os.path.exists(os.path.join(c, "ROWSUM_CLCD_logit.ipynb"))), FIXED)
ORDER = ["logit", "attention_level"]

print("working in:", ROOT)
ok, bad = [], []
for v in ORDER:
    nb = os.path.join(ROOT, "ROWSUM_CLCD_%s.ipynb" % v)
    if not os.path.exists(nb):
        bad.append((v, "notebook not found")); print("SKIP %-16s not found" % v); continue
    print("\n" + "=" * 70); print("RUN  %s" % v); print("=" * 70, flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=3600", nb], capture_output=True, text=True)
    if r.returncode == 0: ok.append(v); print("done in %.0f s" % (time.time() - t0))
    else: bad.append((v, (r.stderr or "")[-1500:])); print("FAILED after %.0f s" % (time.time() - t0))

print("\n" + "=" * 70)
print("succeeded: %s" % (", ".join(ok) or "none"))
for v, why in bad: print("\nFAILED %s\n%s" % (v, why))

R = {}
for v in ORDER:
    p = os.path.join(ROOT, "rowsum_%s.json" % v)
    if os.path.exists(p): R[v] = json.load(open(p, encoding="utf-8"))
if R:
    rows = [("mean |row sum - 1|", "mean_abs_dev", "%.6f"), ("max |row sum - 1|", "max_abs_dev", "%.4f"),
            ("smallest row sum", "min", "%.4f"), ("largest row sum", "max", "%.4f"),
            ("rows within +-0.01 of 1 (%)", "pct_within_0.01", "%.2f"), ("rows within +-0.05 of 1 (%)", "pct_within_0.05", "%.2f"),
            ("rows within +-0.10 of 1 (%)", "pct_within_0.10", "%.2f"),
            ("output change if rows renormalized, before LayerNorm", "rel_change_before_layernorm", "%.5f"),
            ("output change if rows renormalized, after LayerNorm", "rel_change_after_layernorm", "%.5f")]
    print("\n" + "=" * 100)
    print(" Row sums of the attention maps, all blocks, heads and both directions -- CLCD test split")
    print("=" * 100)
    print("%-56s %20s %20s" % ("", "logit (ours)", "attention_level"))
    print("-" * 100)
    for label, key, fmt in rows:
        print("%-56s %20s %20s" % (label, (fmt % R["logit"]["maps"]["all"][key]) if "logit" in R else "-",
                                   (fmt % R["attention_level"]["maps"]["all"][key]) if "attention_level" in R else "-"))
    print("=" * 100)
    for v in R: print(" %s: %d pairs, %d rows, parity %.1e" % (v, R[v]["pairs"], R[v]["maps"]["all"]["rows"], R[v]["parity_max_abs_delta"]))
'''

def cell(kind, text):
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base["outputs"] = []; base["execution_count"] = None
    return base

def source_of(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = re.sub(r'if __name__ == "__main__":(?:\n(?:[ \t].*)?)+',
                 '# (notebook: nothing runs on import)\n', src)
    src = re.sub(r"(?m)^\s*(main|train_model)\s*\(\s*\)\s*$", "", src)
    return src

TE1, TE2, EXT = r"D:\CLCD\test\time1", r"D:\CLCD\test\time2", "png"
VARIANTS = [
 ("logit",           r"BCA_CLCD_518\CLCD_BALANCEDCA_seed_42_518x518",        "Untitled301.ipynb"),
 ("attention_level", r"Ablation_attention\ABL_CLCD_attention_level_seed_42",  "Untitled312.ipynb"),
]
META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}}

if not os.path.isdir(OUTDIR): os.makedirs(OUTDIR)
made = []
for var, rel, nbname in VARIANTS:
    lrun = L(rel)
    assert os.path.exists(os.path.join(lrun, "final", "best_model.pth")), (var, "no checkpoint")
    srcpath = os.path.join(lrun, nbname); assert os.path.exists(srcpath), srcpath
    src = source_of(srcpath)
    scratch = WROOT + "\\ROWSUM\\_scratch_%s" % var
    src, n_ob = re.subn(r'(?m)^OUTPUT_BASE\s*=.*$', lambda m: 'OUTPUT_BASE = r"%s"   # row-sum harness: scratch folder, the run folder is not touched' % scratch, src)
    assert n_ob == 1, (var, "expected exactly one OUTPUT_BASE assignment", n_ob)
    if var == "attention_level":
        assert re.search(r'(?m)^VARIANT\s*=\s*["\']attention_level["\']', src), "the source notebook does not set VARIANT = attention_level"
    body = (EVAL.replace("__VAR__", '"%s"' % var)
                .replace("__CKPT__", 'r"%s"' % W(rel + r"\final\best_model.pth"))
                .replace("__TE1__", 'r"%s"' % TE1).replace("__TE2__", 'r"%s"' % TE2).replace("__EXT__", '"%s"' % EXT)
                .replace("__OUT__", 'r"%s"' % (WROOT + "\\ROWSUM\\rowsum_%s.csv" % var))
                .replace("__OUTJ__", 'r"%s"' % (WROOT + "\\ROWSUM\\rowsum_%s.json" % var)))
    nb = {"cells": [cell("markdown", HOW.format(var=var)), cell("code", src), cell("code", body.lstrip("\n"))],
          "metadata": META, "nbformat": 4, "nbformat_minor": 5}
    outnb = os.path.join(OUTDIR, "ROWSUM_CLCD_%s.ipynb" % var)
    io.open(outnb, "w", encoding="utf-8").write(json.dumps(nb, ensure_ascii=False, indent=1))
    chk = json.load(io.open(outnb, encoding="utf-8"))
    for ci in (1, 2): compile("".join(chk["cells"][ci]["source"]), outnb, "exec")
    assert "if __name__" not in "".join(chk["cells"][1]["source"]), outnb
    made.append(os.path.basename(outnb))
    print("OK   %-16s -> %-34s (both cells compile)" % (var, os.path.basename(outnb)))

run = {"cells": [cell("markdown", RUNALL_MD), cell("code", RUNALL.lstrip("\n"))], "metadata": META, "nbformat": 4, "nbformat_minor": 5}
p = os.path.join(OUTDIR, "RUN_ALL_ROWSUM.ipynb")
io.open(p, "w", encoding="utf-8").write(json.dumps(run, ensure_ascii=False, indent=1))
compile("".join(json.load(io.open(p, encoding="utf-8"))["cells"][1]["source"]), p, "exec")
print("OK   runner           -> RUN_ALL_ROWSUM.ipynb")
print("\n%d notebooks written in %s\\ROWSUM" % (len(made) + 1, WROOT))
