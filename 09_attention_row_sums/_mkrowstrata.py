# -*- coding: utf-8 -*-
# Build ONE notebook that measures, for the attention-level checkpoint, how the
# row sums of the cross-temporal attention maps are distributed -- overall, and
# by the kind of ground each token covers (changed / change boundary / unchanged).
# Inference only.  No training.  Nothing already on disk is overwritten.
#
# The logit-level rule needs no such run: its rows sum to 1 in every stratum by
# construction, which the first measurement already confirmed row by row.
import io, json, re, os

LROOT = os.path.join(os.path.expanduser("~"), "mnt", "Master Thesis Codes", "attention")
WROOT = r"D:\Master Thesis Codes\attention"
OUTDIR = os.path.join(LROOT, "ROWSUM")

def L(rel): return os.path.join(LROOT, *rel.replace("\\", "/").split("/"))
def W(rel): return WROOT + "\\" + rel

EVAL = r'''
# ===================================================================
#  THE NON-UNIT ROW SUMS OF THE ATTENTION-LEVEL RULE, LOOKED AT FOUR WAYS
#  full CLCD test split, inference only, float32, model.eval()
#
#  Same harness as the first row-sum measurement: the attention maps are
#  recomputed from the module's own projections and checked against its real
#  output.  This notebook adds
#
#   1. WHERE   every token is labelled by the ground truth it covers
#              (changed / boundary / unchanged) and the row sums are kept as a
#              histogram, overall and per label;
#   2. WHICH   the same statistics per block, per head and per image, so the
#              result cannot be the work of a few heads or a few images, and
#              the ten largest row sums with where they sit;
#   3. HOW FAR a second pass in which every row of every map is divided by its
#              own sum, to see how much of the change survives to the output of
#              each block, to the fused features and to the predictions.
#
#  The second pass is a probe of the trained model, not a proposal: the model
#  was trained with the un-normalized maps, so its predictions are expected to
#  move when the rows are renormalized.  How far they move is the measurement.
#
#  Row i of S_12 belongs to token i of the first date and row i of S_21 to
#  token i of the second; both dates share the token grid, so one label map
#  serves both directions.
# ===================================================================
import os, csv, json, datetime, numpy as np, torch, torch.nn.functional as F

RS_VARIANT = "attention_level"
CKPT       = __CKPT__
TEST_IMG1  = __TE1__
TEST_IMG2  = __TE2__
TEST_MASK  = __TEM__
IMG_EXT    = __EXT__
OUT_CSV    = __OUT__
OUT_JSON   = __OUTJ__
PREV_JSON  = __PREV__      # the first measurement; a full run must reproduce its overall figure
MAX_PAIRS  = None          # None = full split; set to 4 for a smoke test
TOLS       = (0.01, 0.05, 0.10)
HI         = (1.5, 2.0, 5.0)          # share of rows whose sum exceeds each of these
LO         = (0.75,)                  # share of rows whose sum is below each of these
EDGES      = np.concatenate([[0.0], np.logspace(np.log10(0.45), np.log10(128.0), 601), [1e30]])
STRATA     = ("changed", "boundary", "unchanged_in_pair_with_change", "unchanged_in_pair_without_change")
REPORTED   = dict(f1=81.57, iou=68.88, precision=77.18, recall=86.49, oa=97.09)   # this checkpoint on this split, as tabulated (bfloat16 AMP)

assert os.path.exists(CKPT), "checkpoint not found: %s" % CKPT
_READ  = globals().get("read_image_rasterio") or globals().get("read_image")
_RMASK = globals().get("read_mask_pil") or globals().get("read_mask")
assert _READ is not None, "no image reader found in the source cell"
assert _RMASK is not None, "no mask reader found in the source cell"
THR = float(globals().get("THRESHOLD", 0.5))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

f1s = load_files(TEST_IMG1, IMG_EXT)
f2s = load_files(TEST_IMG2, IMG_EXT)
fms = load_files(TEST_MASK, IMG_EXT)
assert len(f1s) == len(f2s) == len(fms) and len(f1s) > 0, (len(f1s), len(f2s), len(fms))
for a, b, c in zip(f1s, f2s, fms):
    assert os.path.basename(a) == os.path.basename(b) == os.path.basename(c), (a, b, c)
if MAX_PAIRS: f1s, f2s, fms = f1s[:MAX_PAIRS], f2s[:MAX_PAIRS], fms[:MAX_PAIRS]
print("rule: %s | test pairs: %d | threshold %.2f" % (RS_VARIANT, len(f1s), THR))

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
N_HEADS = int(model.cross_blocks[0].attn.num_heads)
print("cross-attention blocks: %d | heads: %d" % (N_BLOCKS, N_HEADS))

G = IMG_SIZE // PATCH_SIZE            # tokens per side
NTOK = G * G

def token_strata(mask_path):
    # Label of every token, in the row-major order the tokens are in, from the mask the model is scored on.
    m = np.asarray(_RMASK(mask_path), dtype=np.float64)                 # (IMG_SIZE, IMG_SIZE), 1 = changed
    assert m.shape == (IMG_SIZE, IMG_SIZE), m.shape
    frac = m.reshape(G, PATCH_SIZE, G, PATCH_SIZE).mean(axis=(1, 3)).reshape(-1)
    lab = np.full(G * G, 2 if m.any() else 3, dtype=np.int64)
    lab[(frac > 0.0) & (frac < 1.0 - 1e-9)] = 1
    lab[frac >= 1.0 - 1e-9] = 0
    return lab, m

def new_acc():
    return dict(hist=np.zeros(len(EDGES) - 1, dtype=np.int64), n=0, sum=0.0, abs=0.0,
                within=[0] * len(TOLS), hi=[0] * len(HI), lo=[0] * len(LO),
                min=float("inf"), max=float("-inf"), pre=0.0, post=0.0, ntok=0)

DIRS   = ("S_12", "S_21")
STAGES = ["block%d" % (b + 1) for b in range(N_BLOCKS)] + ["fusion"]
ACC    = {(b, d, s): new_acc() for b in range(N_BLOCKS) for d in DIRS for s in range(len(STRATA))}
HEADS  = {(b, d): dict(n=0, abs=np.zeros(N_HEADS), w10=np.zeros(N_HEADS), hi2=np.zeros(N_HEADS), mx=np.zeros(N_HEADS))
          for b in range(N_BLOCKS) for d in DIRS}
PROP   = {st: dict(sum=0.0, n=0, by=np.zeros(len(STRATA)), nby=np.zeros(len(STRATA))) for st in STAGES}
CONF   = {"base": np.zeros(4, dtype=np.int64), "renorm": np.zeros(4, dtype=np.int64)}      # TP, FP, FN, TN
PIX    = dict(dp=0.0, n=0, dpmax=0.0, flips=0, flips_changed=0, n_changed=0)
IMGS, TOP = [], []
PARITY, IDENT = [0.0], [None]
CUR  = {"lab": None, "img": None}
MODE = {"renorm": False, "identity": False}
CAP  = {}

def attention_maps(attn, x1, x2):
    # The two attention maps exactly as the attention-level module forms them.
    q1, k1, v1 = attn._split(attn.q1_proj(x1)), attn._split(attn.k1_proj(x1)), attn._split(attn.v1_proj(x1))
    q2, k2, v2 = attn._split(attn.q2_proj(x2)), attn._split(attn.k2_proj(x2)), attn._split(attn.v2_proj(x2))
    N = x1.shape[1]
    bias = attn.relative_bias_table[attn.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
    s12 = (q1 @ k2.transpose(-2, -1)) * attn.scale + bias
    s21 = (q2 @ k1.transpose(-2, -1)) * attn.scale + bias
    a12 = F.softmax(s12, dim=-1)
    a21 = F.softmax(s21, dim=-1)
    S12 = 0.5 * (a12 + a21.transpose(-2, -1))
    S21 = 0.5 * (a21 + a12.transpose(-2, -1))
    return S12, S21, v1, v2

def make_attn_hook(bi):
    def hook(attn, inp, out):
        x1, x2 = inp[0].float(), inp[1].float()
        assert x1.shape[0] == 1 and x1.shape[1] == NTOK, tuple(x1.shape)
        S12, S21, v1, v2 = attention_maps(attn, x1, x2)
        if MODE["renorm"] or MODE["identity"]:
            # replace the module's output: with the same maps (identity, a self-test) or with every row divided by its sum
            outs = []
            for S, v in ((S12, v2), (S21, v1)):
                if MODE["renorm"]: S = S / S.sum(dim=-1, keepdim=True)
                outs.append(attn.out_norm(attn.out_proj(attn._merge(S @ v))).to(out[0].dtype))
            return (outs[0], outs[1])
        lab, im = CUR["lab"], CUR["img"]
        for name, S, v, ref in (("S_12", S12, v2, out[0]), ("S_21", S21, v1, out[1])):
            rs = S.sum(dim=-1, dtype=torch.float64)                      # (1, heads, N)
            y  = attn._merge(S @ v)
            o  = attn.out_norm(attn.out_proj(y))                         # dropout is the identity in eval()
            PARITY[0] = max(PARITY[0], float((o - ref.float()).abs().max().item()))
            yn = attn._merge((S / rs.float().unsqueeze(-1)) @ v)         # every row divided by its own sum
            on = attn.out_norm(attn.out_proj(yn))
            pre  = ((y - yn).norm(dim=-1) / (y.norm(dim=-1) + 1e-12))[0].double().cpu().numpy()   # (N,)
            post = ((o - on).norm(dim=-1) / (o.norm(dim=-1) + 1e-12))[0].double().cpu().numpy()   # (N,)
            r = rs[0].float().cpu().numpy().astype(np.float64)           # (heads, N), float32 values as in the first run
            dall = np.abs(r - 1.0)
            for si in range(len(STRATA)):                                # 1. by what the token covers
                sel = (lab == si)
                if not sel.any(): continue
                a = r[:, sel].reshape(-1); d = np.abs(a - 1.0)
                A = ACC[(bi, name, si)]
                A["hist"] += np.histogram(a, bins=EDGES)[0]
                A["n"] += int(a.size); A["sum"] += float(a.sum()); A["abs"] += float(d.sum())
                for ti, t in enumerate(TOLS): A["within"][ti] += int((d <= t).sum())
                for ti, t in enumerate(HI):   A["hi"][ti] += int((a > t).sum())
                for ti, t in enumerate(LO):   A["lo"][ti] += int((a < t).sum())
                A["min"] = min(A["min"], float(a.min())); A["max"] = max(A["max"], float(a.max()))
                A["pre"] += float(pre[sel].sum()); A["post"] += float(post[sel].sum()); A["ntok"] += int(sel.sum())
            Hh = HEADS[(bi, name)]                                       # 2. per head
            Hh["n"] += int(r.shape[1]); Hh["abs"] += dall.sum(axis=1); Hh["w10"] += (dall <= 0.10).sum(axis=1)
            Hh["hi2"] += (r > 2.0).sum(axis=1); Hh["mx"] = np.maximum(Hh["mx"], r.max(axis=1))
            im["n"] += int(r.size); im["abs"] += float(dall.sum()); im["w10"] += int((dall <= 0.10).sum())   # per image
            im["mx"] = max(im["mx"], float(r.max()))
            flat = r.reshape(-1)                                         # the largest rows, with where they sit
            for j in np.argpartition(flat, -3)[-3:]:
                h, tk = divmod(int(j), r.shape[1])
                TOP.append((float(flat[j]), im["file"], bi + 1, name, h, tk // G, tk % G, STRATA[int(lab[tk])]))
            TOP.sort(key=lambda z: -z[0]); del TOP[10:]
        del S12, S21
        return None
    return hook

def make_block_hook(bi):
    def hook(blk, inp, out):
        CAP["block%d" % (bi + 1)] = (out[0].detach().float(), out[1].detach().float())
    return hook

def fusion_hook(mod, inp, out):
    CAP["fusion"] = (out.detach().float(),)

handles  = [blk.attn.register_forward_hook(make_attn_hook(i)) for i, blk in enumerate(model.cross_blocks)]
handles += [blk.register_forward_hook(make_block_hook(i)) for i, blk in enumerate(model.cross_blocks)]
handles += [model.fusion.register_forward_hook(fusion_hook)]

from tqdm.auto import tqdm
TOK = np.zeros(len(STRATA), dtype=np.int64); PIXF = []
with torch.no_grad():
    for k in tqdm(range(len(f1s)), desc="row sums, two passes per pair"):
        lab, m = token_strata(fms[k]); PIXF.append(float(m.mean()))
        CUR["lab"] = lab
        CUR["img"] = dict(file=os.path.basename(f1s[k]), changed_pct=100.0 * float(m.mean()), n=0, abs=0.0, w10=0, mx=0.0)
        TOK += np.bincount(lab, minlength=len(STRATA))
        x1 = torch.from_numpy(_READ(f1s[k]).transpose(2, 0, 1)[None]).float().to(DEVICE)
        x2 = torch.from_numpy(_READ(f2s[k]).transpose(2, 0, 1)[None]).float().to(DEVICE)

        MODE["renorm"] = False; MODE["identity"] = False; CAP.clear()
        lo0 = model(x1, x2).float(); base = dict(CAP)                    # the model as trained (statistics are taken here)
        if k == 0:                                                       # self-test: replacing the output with itself changes nothing
            MODE["identity"] = True; CAP.clear()
            IDENT[0] = float((model(x1, x2).float() - lo0).abs().max().item()); MODE["identity"] = False
            assert IDENT[0] < 1e-3, "replacing the attention output with itself moved the logits by %.3e -- stop and report" % IDENT[0]
        MODE["renorm"] = True; CAP.clear()
        lo1 = model(x1, x2).float(); ren = dict(CAP); MODE["renorm"] = False     # every row of every map divided by its sum

        for st in STAGES:                                                # 3. how far the change travels
            rel = 0.0
            for za, zb in zip(base[st], ren[st]):
                rel = rel + ((zb - za).norm(dim=-1) / (za.norm(dim=-1) + 1e-12))[0].double().cpu().numpy()
            rel = rel / len(base[st])                                    # (N,), mean over the two branches where there are two
            P = PROP[st]; P["sum"] += float(rel.sum()); P["n"] += int(rel.size)
            P["by"] += np.bincount(lab, weights=rel, minlength=len(STRATA)); P["nby"] += np.bincount(lab, minlength=len(STRATA))
        p0 = torch.sigmoid(lo0[0]).double().cpu().numpy(); p1 = torch.sigmoid(lo1[0]).double().cpu().numpy()
        assert p0.shape == m.shape, (p0.shape, m.shape)
        yt = m > 0.5; b0 = p0 > THR; b1 = p1 > THR
        for key, bb in (("base", b0), ("renorm", b1)):
            CONF[key] += np.array([np.sum(bb & yt), np.sum(bb & ~yt), np.sum(~bb & yt), np.sum(~bb & ~yt)], dtype=np.int64)
        fl = (b0 != b1); dp = np.abs(p1 - p0)
        PIX["dp"] += float(dp.sum()); PIX["n"] += int(dp.size); PIX["dpmax"] = max(PIX["dpmax"], float(dp.max()))
        PIX["flips"] += int(fl.sum()); PIX["flips_changed"] += int((fl & yt).sum()); PIX["n_changed"] += int(yt.sum())
        IMGS.append(CUR["img"])
        del x1, x2, lo0, lo1, base, ren
        CAP.clear()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
for h in handles: h.remove()

assert PARITY[0] < 1e-3, ("the recomputed maps do not reproduce the module output: max|delta|=%.3e "
                          "-- stop, the harness is wrong, not the result" % PARITY[0])
print("\nparity  max|recomputed - module output| = %.3e   (must be ~0)" % PARITY[0])
print("self-test: replacing the attention output with itself moved the logits by %.3e   (must be ~0)" % IDENT[0])

def merge(keys):
    out = new_acc()
    for k in keys:
        A = ACC[k]
        out["hist"] += A["hist"]; out["n"] += A["n"]; out["sum"] += A["sum"]; out["abs"] += A["abs"]
        for f in ("within", "hi", "lo"): out[f] = [x + y for x, y in zip(out[f], A[f])]
        out["min"] = min(out["min"], A["min"]); out["max"] = max(out["max"], A["max"])
        out["pre"] += A["pre"]; out["post"] += A["post"]; out["ntok"] += A["ntok"]
    return out

def quantile(hist, q):
    # Quantile read off the histogram, log-interpolated inside its bin (bins are 0.95% wide).
    c = np.cumsum(hist); tot = int(c[-1])
    if tot == 0: return float("nan")
    k = int(np.searchsorted(c, q * tot, side="left"))
    lo, hi = float(EDGES[k]), float(EDGES[k + 1])
    before = int(c[k - 1]) if k > 0 else 0
    f = (q * tot - before) / max(int(hist[k]), 1)
    return float(lo * (hi / lo) ** f) if (lo > 0 and hi < 1e29) else float(lo)

def summarize(A, total_rows):
    if A["n"] == 0: return None
    out = dict(rows=int(A["n"]), share_of_rows_pct=100.0 * A["n"] / total_rows, mean=A["sum"] / A["n"],
               mean_abs_dev=A["abs"] / A["n"], min=A["min"], max=A["max"],
               p05=quantile(A["hist"], 0.05), p25=quantile(A["hist"], 0.25), p50=quantile(A["hist"], 0.50),
               p75=quantile(A["hist"], 0.75), p95=quantile(A["hist"], 0.95), p99=quantile(A["hist"], 0.99))
    for ti, t in enumerate(TOLS): out["pct_within_%.2f" % t] = 100.0 * A["within"][ti] / A["n"]
    for ti, t in enumerate(HI):   out["pct_above_%.2f" % t]  = 100.0 * A["hi"][ti] / A["n"]
    for ti, t in enumerate(LO):   out["pct_below_%.2f" % t]  = 100.0 * A["lo"][ti] / A["n"]
    out["rel_change_before_layernorm"] = A["pre"] / max(A["ntok"], 1)
    out["rel_change_after_layernorm"]  = A["post"] / max(A["ntok"], 1)
    return out

def prf(c):
    tp, fp, fn, tn = [float(v) for v in c]
    return dict(f1=100.0 * 2 * tp / max(2 * tp + fp + fn, 1.0), iou=100.0 * tp / max(tp + fp + fn, 1.0),
                precision=100.0 * tp / max(tp + fp, 1.0), recall=100.0 * tp / max(tp + fn, 1.0),
                oa=100.0 * (tp + tn) / max(tp + fp + fn + tn, 1.0))

def spread(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(n=int(v.size), min=float(v.min()), p25=float(np.percentile(v, 25)), median=float(np.median(v)),
                p75=float(np.percentile(v, 75)), max=float(v.max()), mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0)

ALLK = list(ACC.keys())
TOTAL = merge(ALLK); NROWS = TOTAL["n"]
GROUPS = {"changed": (0,), "boundary": (1,), "unchanged": (2, 3),
          "unchanged_in_pair_with_change": (2,), "unchanged_in_pair_without_change": (3,)}
res = {"all": summarize(TOTAL, NROWS), "by_stratum": {}, "by_stratum_and_direction": {}, "by_stratum_and_block": {}, "by_block_and_direction": {}}
hist = {"all": TOTAL["hist"].tolist()}
for g, ss in GROUPS.items():
    A = merge([k for k in ALLK if k[2] in ss])
    res["by_stratum"][g] = summarize(A, NROWS); hist[g] = A["hist"].tolist()
    for d in DIRS:
        res["by_stratum_and_direction"]["%s|%s" % (g, d)] = summarize(merge([k for k in ALLK if k[2] in ss and k[1] == d]), NROWS)
    for b in range(N_BLOCKS):
        res["by_stratum_and_block"]["%s|block%d" % (g, b + 1)] = summarize(merge([k for k in ALLK if k[2] in ss and k[0] == b]), NROWS)
for b in range(N_BLOCKS):
    for d in DIRS:
        res["by_block_and_direction"]["block%d|%s" % (b + 1, d)] = summarize(merge([k for k in ALLK if k[0] == b and k[1] == d]), NROWS)

heads = []
for (b, d), Hh in HEADS.items():
    for h in range(N_HEADS):
        heads.append(dict(block=b + 1, direction=d, head=h, rows=int(Hh["n"]), mean_abs_dev=float(Hh["abs"][h] / Hh["n"]),
                          pct_within_0_10=float(100.0 * Hh["w10"][h] / Hh["n"]), pct_above_2=float(100.0 * Hh["hi2"][h] / Hh["n"]),
                          max=float(Hh["mx"][h])))
images = [dict(file=im["file"], changed_pct=im["changed_pct"], rows=int(im["n"]), mean_abs_dev=im["abs"] / im["n"],
               pct_within_0_10=100.0 * im["w10"] / im["n"], max=im["mx"]) for im in IMGS]
top = [dict(row_sum=z[0], received_attention=2.0 * z[0] - 1.0, share_of_all_attention_pct=100.0 * (2.0 * z[0] - 1.0) / NTOK,
            file=z[1], block=z[2], direction=z[3], head=z[4], token_row=z[5], token_col=z[6], covers=z[7]) for z in TOP]
prop = {}
for st in STAGES:
    P = PROP[st]
    prop[st] = dict(all=P["sum"] / max(P["n"], 1),
                    **{g: float(sum(P["by"][i] for i in ss) / max(sum(P["nby"][i] for i in ss), 1)) for g, ss in GROUPS.items()})
pred = dict(base=prf(CONF["base"]), renormalized=prf(CONF["renorm"]), reported=REPORTED,
            confusion_base=CONF["base"].tolist(), confusion_renormalized=CONF["renorm"].tolist(),
            mean_abs_prob_change=PIX["dp"] / max(PIX["n"], 1), max_abs_prob_change=PIX["dpmax"],
            pixels_flipped_pct=100.0 * PIX["flips"] / max(PIX["n"], 1),
            changed_pixels_flipped_pct=100.0 * PIX["flips_changed"] / max(PIX["n_changed"], 1),
            unchanged_pixels_flipped_pct=100.0 * (PIX["flips"] - PIX["flips_changed"]) / max(PIX["n"] - PIX["n_changed"], 1))

tokens = {STRATA[i]: int(TOK[i]) for i in range(len(STRATA))}
meta = dict(rule=RS_VARIANT, pairs=len(f1s), blocks=N_BLOCKS, heads_per_block=N_HEADS, precision="float32", parity_max_abs_delta=PARITY[0],
            identity_replacement_max_abs_logit_delta=IDENT[0], threshold=THR,
            checkpoint=CKPT, tokens_per_side=G, tokens=tokens, tokens_total=int(TOK.sum()),
            changed_pixel_pct=100.0 * float(np.mean(PIXF)), pairs_with_change=int(sum(1 for p in PIXF if p > 0)))
summary = dict(per_head_mean_abs_dev=spread([h["mean_abs_dev"] for h in heads]),
               per_head_pct_within_0_10=spread([h["pct_within_0_10"] for h in heads]),
               head_maps_total=len(heads), head_maps_with_mean_abs_dev_above_0_05=int(sum(1 for h in heads if h["mean_abs_dev"] > 0.05)),
               head_maps_with_mean_abs_dev_above_0_10=int(sum(1 for h in heads if h["mean_abs_dev"] > 0.10)),
               per_image_mean_abs_dev=spread([i["mean_abs_dev"] for i in images]),
               per_image_pct_within_0_10=spread([i["pct_within_0_10"] for i in images]),
               per_image_max=spread([i["max"] for i in images]))

W1 = "%-34s %7s %9s %8s %8s %8s %8s %9s %10s %9s %9s %10s %10s"
print("\n" + "=" * 150)
print(" 1. ROW SUMS BY WHAT THE TOKEN COVERS -- rule: %s -- %d pairs (%d with change), changed pixels %.2f%%"
      % (RS_VARIANT, len(f1s), meta["pairs_with_change"], meta["changed_pixel_pct"]))
print("=" * 150)
print(W1 % ("tokens", "rows %", "mean|s-1|", "mean s", "p05", "median", "p95", "p99", "in+-0.10%", "s>1.5 %", "s>2 %", "rel preLN", "rel postLN"))
print("-" * 150)
for g in ["all"] + list(GROUPS.keys()):
    mm = res["all"] if g == "all" else res["by_stratum"][g]
    if mm is None: print("%-34s (no tokens)" % g); continue
    print(W1 % (g, "%.2f" % mm["share_of_rows_pct"], "%.4f" % mm["mean_abs_dev"], "%.4f" % mm["mean"], "%.3f" % mm["p05"],
                "%.3f" % mm["p50"], "%.3f" % mm["p95"], "%.3f" % mm["p99"], "%.2f" % mm["pct_within_0.10"],
                "%.2f" % mm["pct_above_1.50"], "%.2f" % mm["pct_above_2.00"],
                "%.4f" % mm["rel_change_before_layernorm"], "%.4f" % mm["rel_change_after_layernorm"]))
print("=" * 150)
print(" s = sum of one row of an attention map (one query token, one head). %d rows in all." % NROWS)
print(" Row i sums to (1 + c_i)/2, c_i being the attention that token i RECEIVES from the other date:")
print(" a mean s above 1 in a stratum means its tokens are attended to more than average.")

sh, si_ = summary["per_head_mean_abs_dev"], summary["per_image_mean_abs_dev"]
print("\n 2. NOT A FEW HEADS, NOT A FEW IMAGES")
print("    mean|s-1| over the %d head maps (block x direction x head): min %.4f  median %.4f  max %.4f ; above 0.05 in %d, above 0.10 in %d"
      % (sh["n"], sh["min"], sh["median"], sh["max"], summary["head_maps_with_mean_abs_dev_above_0_05"], summary["head_maps_with_mean_abs_dev_above_0_10"]))
print("    mean|s-1| over the %d test pairs:                              min %.4f  median %.4f  max %.4f  (std %.4f)"
      % (si_["n"], si_["min"], si_["median"], si_["max"], si_["std"]))
print("    the ten largest row sums (c = attention received = 2s - 1; N = %d tokens):" % NTOK)
for z in top:
    print("      s=%8.3f  c=%8.2f (%.1f%% of the other date's attention in that head)  %s  block %d %s head %2d  token (%d,%d)  %s"
          % (z["row_sum"], z["received_attention"], z["share_of_all_attention_pct"], z["file"], z["block"], z["direction"], z["head"],
             z["token_row"], z["token_col"], z["covers"]))

print("\n 3. HOW FAR THE CHANGE TRAVELS WHEN EVERY ROW IS DIVIDED BY ITS SUM (mean relative change per token, l2)")
print("    %-10s %10s %10s %10s %10s" % ("after", "all", "changed", "boundary", "unchanged"))
for st in STAGES:
    print("    %-10s %10.4f %10.4f %10.4f %10.4f" % (st, prop[st]["all"], prop[st]["changed"], prop[st]["boundary"], prop[st]["unchanged"]))
print("    predictions: mean |dp| %.5f, max |dp| %.4f, pixels whose label flips %.3f%% (changed pixels %.3f%%, unchanged %.3f%%)"
      % (pred["mean_abs_prob_change"], pred["max_abs_prob_change"], pred["pixels_flipped_pct"], pred["changed_pixels_flipped_pct"], pred["unchanged_pixels_flipped_pct"]))
print("    %-22s %8s %8s %10s %8s %8s" % ("", "F1", "IoU", "Precision", "Recall", "OA"))
for nm, key in (("as trained (this run)", "base"), ("rows renormalized", "renormalized"), ("as tabulated", "reported")):
    q = pred[key]; print("    %-22s %8.2f %8.2f %10.2f %8.2f %8.2f" % (nm, q["f1"], q["iou"], q["precision"], q["recall"], q["oa"]))
print("    (this run is float32; the tabulated figures were scored under bfloat16 AMP, so the first and third rows may differ in the second decimal)")

if os.path.exists(PREV_JSON):
    prev = json.load(open(PREV_JSON, encoding="utf-8"))["maps"]["all"]
    d1 = abs(res["all"]["mean_abs_dev"] - prev["mean_abs_dev"])
    print("\ncheck against the first measurement: mean|s-1| %.6f vs %.6f (diff %.1e), rows %d vs %d"
          % (res["all"]["mean_abs_dev"], prev["mean_abs_dev"], d1, NROWS, prev["rows"]))
    if not MAX_PAIRS:
        assert NROWS == prev["rows"] and d1 < 1e-5, "this run does not reproduce the first measurement -- stop and report"
    meta["reproduces_first_measurement"] = bool(NROWS == prev["rows"] and d1 < 1e-5)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
cols = ["rows", "share_of_rows_pct", "mean", "mean_abs_dev", "min", "p05", "p25", "p50", "p75", "p95", "p99", "max"] + \
       ["pct_within_%.2f" % t for t in TOLS] + ["pct_above_%.2f" % t for t in HI] + ["pct_below_%.2f" % t for t in LO] + \
       ["rel_change_before_layernorm", "rel_change_after_layernorm"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["rule", "group"] + cols)
    rowsets = [("all", res["all"])] + [(g, mm) for g, mm in res["by_stratum"].items()] + \
              list(res["by_block_and_direction"].items()) + list(res["by_stratum_and_direction"].items()) + list(res["by_stratum_and_block"].items())
    for g, mm in rowsets:
        if mm is None: continue
        w.writerow([RS_VARIANT, g] + [mm[c] if c == "rows" else "%.8f" % mm[c] for c in cols])
    for k2, v2 in meta.items():
        if not isinstance(v2, dict): w.writerow([RS_VARIANT, "_meta_" + k2, v2])
    w.writerow([RS_VARIANT, "_meta_written_utc", datetime.datetime.utcnow().isoformat(timespec="seconds")])
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(dict(meta, strata=res, summary=summary, heads=heads, images=images, top_rows=top, propagation=prop,
                   predictions=pred, histogram=dict(edges=EDGES.tolist(), counts=hist)), fh, indent=1)
print("written ->", OUT_CSV)
print("written ->", OUT_JSON)
'''

HOW = """# The non-unit row sums of the attention-level rule: where, in which heads and images, and how far they travel

Run the two cells in order (or Run All). Inference only, no training, about
six minutes on the GPU (two passes per pair). The second cell checks that the
recomputed attention maps reproduce the model's own output, that replacing an
attention output with itself changes nothing, and that the overall figure
reproduces the first measurement. If an assertion fires, stop and report it -
the harness is wrong, not the result.

Writes `rowstrata_attention_level.csv` / `.json` in
`D:\\Master Thesis Codes\\attention\\ROWSUM`.
"""

RUNALL_MD = """# Run the stratified row-sum notebook

One cell. It executes `ROWSTRATA_CLCD_attention_level.ipynb` from
`D:\\Master Thesis Codes\\attention\\ROWSUM` in this Python environment and prints
the result. It works from any folder.
"""

RUNALL = r'''
import os, sys, subprocess, time, json

ROOT = r"D:\Master Thesis Codes\attention\ROWSUM"
nb   = os.path.join(ROOT, "ROWSTRATA_CLCD_attention_level.ipynb")
assert os.path.exists(nb), "notebook not found: %s" % nb
print("running:", nb, "(about six minutes)", flush=True)
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=7200", nb], capture_output=True, text=True)
if r.returncode != 0:
    print("FAILED after %.0f s\n" % (time.time() - t0)); print((r.stderr or "")[-4000:])
else:
    print("done in %.0f s\n" % (time.time() - t0))
    out = json.load(open(nb, encoding="utf-8"))["cells"][2].get("outputs", [])
    txt = "".join("".join(o.get("text", [])) for o in out if o.get("output_type") == "stream" and o.get("name") == "stdout")
    k = txt.find("parity"); print(txt[k:] if k >= 0 else txt[-6000:])
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

TE1, TE2, TEM, EXT = r"D:\CLCD\test\time1", r"D:\CLCD\test\time2", r"D:\CLCD\test\label", "png"
var, rel, nbname = "attention_level", r"Ablation_attention\ABL_CLCD_attention_level_seed_42", "Untitled312.ipynb"
META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}}

lrun = L(rel)
assert os.path.exists(os.path.join(lrun, "final", "best_model.pth")), "no checkpoint"
srcpath = os.path.join(lrun, nbname); assert os.path.exists(srcpath), srcpath
src = source_of(srcpath)
assert re.search(r'(?m)^VARIANT\s*=\s*["\']attention_level["\']', src), "the source notebook does not set VARIANT = attention_level"
scratch = WROOT + "\\ROWSUM\\_scratch_%s" % var
src, n_ob = re.subn(r'(?m)^OUTPUT_BASE\s*=.*$', lambda m: 'OUTPUT_BASE = r"%s"   # row-sum harness: scratch folder, the run folder is not touched' % scratch, src)
assert n_ob == 1, ("expected exactly one OUTPUT_BASE assignment", n_ob)
body = (EVAL.replace("__CKPT__", 'r"%s"' % W(rel + r"\final\best_model.pth"))
            .replace("__TE1__", 'r"%s"' % TE1).replace("__TE2__", 'r"%s"' % TE2).replace("__TEM__", 'r"%s"' % TEM)
            .replace("__EXT__", '"%s"' % EXT)
            .replace("__OUT__", 'r"%s"' % (WROOT + "\\ROWSUM\\rowstrata_%s.csv" % var))
            .replace("__OUTJ__", 'r"%s"' % (WROOT + "\\ROWSUM\\rowstrata_%s.json" % var))
            .replace("__PREV__", 'r"%s"' % (WROOT + "\\ROWSUM\\rowsum_%s.json" % var)))
assert "__" not in re.sub(r"__name__|__init__|__getitem__|__len__", "", body), "unfilled placeholder"
nb = {"cells": [cell("markdown", HOW), cell("code", src), cell("code", body.lstrip("\n"))],
      "metadata": META, "nbformat": 4, "nbformat_minor": 5}
outnb = os.path.join(OUTDIR, "ROWSTRATA_CLCD_%s.ipynb" % var)
assert not os.path.exists(outnb) or os.environ.get("ROWSTRATA_REGEN") == "1", "already generated; set ROWSTRATA_REGEN=1 to rebuild"
io.open(outnb, "w", encoding="utf-8").write(json.dumps(nb, ensure_ascii=False, indent=1))
chk = json.load(io.open(outnb, encoding="utf-8"))
for ci in (1, 2): compile("".join(chk["cells"][ci]["source"]), outnb, "exec")
assert "if __name__" not in "".join(chk["cells"][1]["source"]), outnb
print("OK   %s (both cells compile)" % os.path.basename(outnb))

run = {"cells": [cell("markdown", RUNALL_MD), cell("code", RUNALL.lstrip("\n"))], "metadata": META, "nbformat": 4, "nbformat_minor": 5}
p = os.path.join(OUTDIR, "RUN_ROWSTRATA.ipynb")
io.open(p, "w", encoding="utf-8").write(json.dumps(run, ensure_ascii=False, indent=1))
compile("".join(json.load(io.open(p, encoding="utf-8"))["cells"][1]["source"]), p, "exec")
print("OK   RUN_ROWSTRATA.ipynb")
