"""
Report DGATE's model complexity: parameters, FLOPs, latency and peak memory.

The Discussion states that the design "trades computational compactness for the
transferable representational capacity of a large pretrained encoder, which
motivates reporting model complexity transparently rather than competing on
efficiency". This script produces the numbers that sentence promises and that
the paper otherwise never gives.

    python make_complexity.py                 # uses the seed-42 518 checkpoint
    python make_complexity.py --no-ckpt       # architecture only, no weights needed
    python make_complexity.py --runs 50

Nothing here depends on the trained weights: parameter counts and FLOPs are a
property of the architecture, and latency is measured on zero tensors. --no-ckpt
therefore gives identical numbers without loading the 1.98 GB checkpoint.

Three conventions worth stating in the paper, because reviewers ask:
  * FLOPs are 2 x MACs, counted by torch.utils.flop_counter over ONE forward
    pass on ONE bi-temporal pair -- that is, both encoder branches included.
  * Latency is measured after warm-up with CUDA synchronisation, so it excludes
    kernel autotuning but includes the full encode-fuse-decode path.
  * Peak memory is torch.cuda.max_memory_allocated during the timed passes.

Writes complexity.txt next to the paper, ending with a sentence ready to paste
into Section 4.2.
"""
import argparse
import contextlib
import re
import statistics
import time
import math
import os
import sys
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from transformers import Dinov2Model


# ─── configuration (matches the Hi-CNA 518 runs) ──────────────────────
IMG_SIZE     = 518
PATCH_SIZE   = 14
NUM_CHANNELS = 4
EMBED_DIM    = 768
NUM_HEADS    = 12
DROPOUT      = 0.26943406970601036      # must match the trained checkpoint
NUM_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2
THRESHOLD    = 0.5
NBINS        = 1 << 20                  # probability-histogram resolution

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)


# ─── model (identical to the trained architecture) ────────────────────
class DropPath(nn.Module):
    def __init__(self, p=0.1):
        super().__init__(); self.p = p
    def forward(self, x):
        return x                       # inference only


class BalancedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=NUM_HEADS, dropout=DROPOUT, num_patches=NUM_PATCHES):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        for n in ["q1", "k1", "v1", "q2", "k2", "v2"]:
            setattr(self, f"{n}_proj", nn.Linear(dim, dim))
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        gs = int(math.sqrt(num_patches))
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * gs - 1) ** 2, num_heads))
        c = torch.arange(gs)
        grid = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        rc = grid[:, None, :] - grid[None, :, :]
        rc[..., 0] += gs - 1; rc[..., 1] += gs - 1
        rc[..., 0] *= 2 * (gs - 1) + 1
        self.register_buffer("rel_index", rc.sum(-1).long())

    def _split(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge(self, x):
        return x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], -1)

    def forward(self, x1, x2):
        q1, k1, v1 = self._split(self.q1_proj(x1)), self._split(self.k1_proj(x1)), self._split(self.v1_proj(x1))
        q2, k2, v2 = self._split(self.q2_proj(x2)), self._split(self.k2_proj(x2)), self._split(self.v2_proj(x2))
        N = x1.shape[1]
        bias = self.relative_bias_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        s12 = (q1 @ k2.transpose(-2, -1)) * self.scale + bias
        s21 = (q2 @ k1.transpose(-2, -1)) * self.scale + bias
        M = (s12 + s21.transpose(-2, -1)) / 2.0
        S12 = F.softmax(M, dim=-1).type_as(x1)
        S21 = F.softmax(M.transpose(-2, -1), dim=-1).type_as(x2)
        o1 = self._merge(S12 @ v2)
        o2 = self._merge(S21 @ v1)
        return (self.dropout(self.out_norm(self.out_proj(o1))),
                self.dropout(self.out_norm(self.out_proj(o2))))


class TransformerBlock(nn.Module):
    def __init__(self, dim=EMBED_DIM, nh=NUM_HEADS, mlp_ratio=4.0,
                 dropout=DROPOUT, drop_path=0.1, num_patches=NUM_PATCHES):
        super().__init__()
        self.norm1_x1 = nn.LayerNorm(dim); self.norm1_x2 = nn.LayerNorm(dim)
        self.attn = BalancedCrossAttention(dim, nh, dropout, num_patches)
        self.norm2_x1 = nn.LayerNorm(dim); self.norm2_x2 = nn.LayerNorm(dim)
        md = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, md), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(md, dim), nn.Dropout(dropout))
        self.dp = DropPath(drop_path)

    def forward(self, x1, x2):
        a1, a2 = self.attn(self.norm1_x1(x1), self.norm1_x2(x2))
        x1 = x1 + self.dp(a1); x2 = x2 + self.dp(a2)
        x1 = x1 + self.dp(self.mlp(self.norm2_x1(x1)))
        x2 = x2 + self.dp(self.mlp(self.norm2_x2(x2)))
        return x1, x2


class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_chans=NUM_CHANNELS):
        super().__init__()
        try:
            self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", disable_mmap=True)
        except TypeError:
            self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", low_cpu_mem_usage=False)
        old = self.vit.embeddings.patch_embeddings.projection
        new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=old.bias is not None)
        new.weight.data[:, :3] = old.weight.data
        new.weight.data[:, 3:4] = old.weight.data.mean(dim=1, keepdim=True).clone()
        if old.bias is not None:
            new.bias.data = old.bias.data.clone()
        self.vit.embeddings.patch_embeddings.projection = new
        self.vit.config.num_channels = in_chans
        self.vit.embeddings.patch_embeddings.num_channels = in_chans
        self.embed_dim = self.vit.config.hidden_size
        self.num_patches = NUM_PATCHES

    def forward(self, x):
        out = self.vit(pixel_values=x, output_hidden_states=True, interpolate_pos_encoding=True)
        hs = out.hidden_states
        H = W = IMG_SIZE // PATCH_SIZE
        to_s = lambda h: h[:, 1:].transpose(1, 2).reshape(x.size(0), self.embed_dim, H, W)
        return hs[-1][:, 1:], [to_s(hs[i]) for i in [0, 4, 8, 12]]


class SiameseViT(nn.Module):
    def __init__(self, in_chans=NUM_CHANNELS, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, mlp_ratio=4.0, dropout=DROPOUT):
        super().__init__()
        self.encoder = VisionTransformerEncoder(in_chans)
        np_ = self.encoder.num_patches
        self.cross_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout,
                             drop_path=0.1 * i / max(3, 1), num_patches=np_)
            for i in range(4)])
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim))
        self.skip_proj0 = nn.Conv2d(embed_dim * 3, embed_dim * 2, 1)
        self.skip_proj1 = nn.Conv2d(embed_dim * 2 + embed_dim * 2, embed_dim, 1)
        self.skip_proj2 = nn.Conv2d(embed_dim + embed_dim * 2, embed_dim // 2, 1)
        self.skip_proj3 = nn.Conv2d(embed_dim // 2 + embed_dim * 2, embed_dim // 4, 1)

        def dec(ic, oc):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc),
                                 nn.ReLU(inplace=True),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        self.decoder = nn.ModuleList([
            dec(embed_dim * 2, embed_dim * 2), dec(embed_dim, embed_dim),
            dec(embed_dim // 2, embed_dim // 2), dec(embed_dim // 4, embed_dim // 4),
            nn.Sequential(nn.Conv2d(embed_dim // 4, embed_dim // 8, 3, padding=1),
                          nn.BatchNorm2d(embed_dim // 8), nn.ReLU(inplace=True),
                          nn.Dropout2d(dropout), nn.Conv2d(embed_dim // 8, 1, 1))])

    def forward(self, i1, i2):
        b = i1.shape[0]
        x1, f1 = self.encoder(i1)
        x2, f2 = self.encoder(i2)
        for blk in self.cross_blocks:
            x1, x2 = blk(x1, x2)
        x = self.fusion(torch.cat([x1, x2, torch.abs(x1 - x2)], dim=-1))
        H = W = IMG_SIZE // PATCH_SIZE
        x = x.transpose(1, 2).reshape(b, EMBED_DIM, H, W)
        x = self.skip_proj0(torch.cat([x, torch.cat([f1[0], f2[0]], 1)], 1))
        x = self.decoder[0](x)
        for i in range(1, 4):
            sk = F.interpolate(torch.cat([f1[i], f2[i]], 1), size=x.shape[2:],
                               mode="bilinear", align_corners=False)
            x = getattr(self, f"skip_proj{i}")(torch.cat([x, sk], 1))
            x = self.decoder[i](x)
        out = self.decoder[4](x)
        return F.interpolate(out, size=(IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False).squeeze(1)



# ─── configuration (edit when running inside a notebook) ──────────────
CKPT    = r"D:\Master Thesis Codes\attention\BCA_HICNA_518\BALANCEDCA_seed_42_FIXED_518\final\best_model.pth"
OUTDIR  = r"D:\Master Thesis Codes\Research__MS___Ahmed_Ali_Metwally___Paper2_2026 (1)"
BATCHES = (1, 4)        # batch sizes to time
WARMUP  = 5             # untimed forward passes before measuring
RUNS    = 20            # timed forward passes per batch size
UNFREEZE_FROM = 6       # encoder layers 0..5 stay frozen; 6..11 unfreeze in Stage 2


def human(n):
    return f"{n/1e6:.2f} M" if n < 1e9 else f"{n/1e9:.3f} B"


def group_of(name):
    """Which part of DGATE a parameter belongs to."""
    if name.startswith("encoder."):
        # encoder.vit.encoder.layer.<i>. ...
        m = re.search(r"\.layer\.(\d+)\.", name)
        if m:
            i = int(m.group(1))
            return "encoder: frozen layers 1-6" if i < UNFREEZE_FROM \
                   else "encoder: unfrozen layers 7-12"
        if "patch_embeddings" in name:
            return "encoder: patch embedding (4-channel)"
        return "encoder: embeddings / final norm"
    if name.startswith("cross_blocks."):
        return "Balanced Cross-Attention blocks (x4)"
    if name.startswith("fusion."):
        return "fusion head"
    if name.startswith("skip_proj"):
        return "decoder: skip projections"
    if name.startswith("decoder."):
        return "decoder: upsampling stages"
    return "other"


def count_params(model):
    groups, total, trainable = {}, 0, 0
    for n, p in model.named_parameters():
        g = group_of(n)
        groups[g] = groups.get(g, 0) + p.numel()
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return groups, total, trainable


def measure_flops(model, batch):
    """MACs->FLOPs for one forward pass, via PyTorch's own counter (torch>=2.0).

    FlopCounterMode reports 2*MACs for matmul and conv, which is the convention
    most papers use when they quote GFLOPs. Returns None if unavailable.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    a = torch.zeros(batch, NUM_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    b = torch.zeros_like(a)
    counter = FlopCounterMode(display=False)
    try:
        with counter, torch.no_grad():
            model(a, b)
        return counter.get_total_flops()
    except Exception as e:
        print(f"  (FLOP counting failed: {type(e).__name__}: {e})")
        return None
    finally:
        del a, b
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def measure_latency(model, batch, warmup, runs, amp):
    a = torch.zeros(batch, NUM_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE)
    b = torch.zeros_like(a)
    ctx = (lambda: autocast(device_type=DEVICE.type, dtype=AMP_DTYPE)) if amp \
          else (lambda: contextlib.nullcontext())
    with torch.no_grad():
        for _ in range(warmup):
            with ctx():
                model(a, b)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            with ctx():
                model(a, b)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else float("nan")
    del a, b
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT,
                    help="optional; parameter counts and FLOPs do not depend on the weights")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--no-ckpt", action="store_true",
                    help="build the architecture without loading the checkpoint")
    # parse_known_args: Jupyter injects "-f <kernel.json>" into sys.argv.
    args, _ = ap.parse_known_args()

    print(f"device {DEVICE} | resolution {IMG_SIZE} | {NUM_PATCHES} tokens per branch\n")
    torch.manual_seed(42)
    model = SiameseViT(dropout=DROPOUT).to(DEVICE).eval()

    if not args.no_ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model_state_dict", ck))
        print(f"loaded {os.path.basename(args.ckpt)}\n")
        del ck
    else:
        print("running on the freshly built architecture (weights do not affect these numbers)\n")

    # ---- parameters
    groups, total, trainable = count_params(model)
    print("=" * 68)
    print("PARAMETERS")
    print("=" * 68)
    for g in sorted(groups, key=lambda k: -groups[k]):
        print(f"  {g:<40} {human(groups[g]):>10}  ({100*groups[g]/total:5.1f}%)")
    print(f"  {'-'*40} {'-'*10}")
    print(f"  {'TOTAL':<40} {human(total):>10}")
    frozen = groups.get("encoder: frozen layers 1-6", 0)
    print(f"  {'Stage 1 trainable (backbone frozen)':<40} {human(total - sum(v for k,v in groups.items() if k.startswith('encoder:'))):>10}")
    print(f"  {'Stage 2 trainable (layers 7-12 added)':<40} {human(total - frozen):>10}")
    print(f"  {'frozen throughout':<40} {human(frozen):>10}")

    # ---- FLOPs
    print("\n" + "=" * 68)
    print("COMPUTE  (one bi-temporal pair, both branches, batch = 1)")
    print("=" * 68)
    f1 = measure_flops(model, 1)
    if f1:
        print(f"  forward FLOPs                            {f1/1e9:8.1f} G")
        print(f"  (2 x MACs, the convention used when papers quote GFLOPs)")
    else:
        print("  FLOP counter unavailable (needs torch >= 2.0)")

    # ---- latency
    print("\n" + "=" * 68)
    print(f"LATENCY  ({args.warmup} warm-up + {args.runs} timed passes)")
    print("=" * 68)
    rows = []
    for amp in ([False, True] if DEVICE.type == "cuda" else [False]):
        tag = f"AMP {AMP_DTYPE}".replace("torch.", "") if amp else "fp32"
        for bs in BATCHES:
            try:
                mu, sd, peak = measure_latency(model, bs, args.warmup, args.runs, amp)
            except RuntimeError as e:
                if 'out of memory' not in str(e).lower():
                    raise
                print(f"  {tag:<12} batch {bs:<3} out of memory")
                torch.cuda.empty_cache()
                continue
            per_img = mu / bs
            rows.append((tag, bs, mu, sd, per_img, peak))
            print(f"  {tag:<12} batch {bs:<3} {mu:7.1f} +/- {sd:4.1f} ms"
                  f"   {per_img:6.1f} ms/pair   {1000/per_img:5.2f} pairs/s"
                  f"   peak {peak:.2f} GiB")

    # ---- ready-to-paste LaTeX
    os.makedirs(args.outdir, exist_ok=True)
    dst = os.path.join(args.outdir, "complexity.txt")
    best = min(rows, key=lambda r: r[4]) if rows else None
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(f"device: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'}\n")
        fh.write(f"resolution: {IMG_SIZE}x{IMG_SIZE}  tokens/branch: {NUM_PATCHES}\n\n")
        for g in sorted(groups, key=lambda k: -groups[k]):
            fh.write(f"{g:<42}{groups[g]}\n")
        fh.write(f"{'TOTAL':<42}{total}\n")
        fh.write(f"{'frozen throughout':<42}{frozen}\n")
        if f1:
            fh.write(f"\nforward GFLOPs (batch 1): {f1/1e9:.1f}\n")
        for tag, bs, mu, sd, per_img, peak in rows:
            fh.write(f"latency {tag} batch {bs}: {mu:.1f} +/- {sd:.1f} ms "
                     f"({per_img:.1f} ms/pair, peak {peak:.2f} GiB)\n")

        fh.write("\n--- LaTeX sentence for Section 4.2 ---\n")
        s = (f"DGATE has {total/1e6:.1f}\\,M parameters in total, of which "
             f"{frozen/1e6:.1f}\\,M (encoder layers $\\ell_{{1\\text{{--}}6}}$) remain frozen "
             f"throughout and {(total-frozen)/1e6:.1f}\\,M are trainable in Stage~2.")
        if f1:
            s += (f" A forward pass over one bi-temporal pair at $518 \\times 518$ costs "
                  f"{f1/1e9:.0f}\\,GFLOPs")
            if best:
                s += (f" and takes {best[4]:.0f}\\,ms on a "
                      f"{torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'} "
                      f"({best[0]}, batch~{best[1]}), with a peak memory of {best[5]:.1f}\\,GiB")
            s += ("."
                  " We report these figures because the model trades computational compactness"
                  " for the representational capacity of a large pretrained encoder, and that"
                  " trade-off should be visible rather than implied.")
        fh.write(s + "\n")
    print(f"\nwrote {dst}")
    print("It ends with a ready-to-paste sentence for Section 4.2.")


# Running as a script and running inside a notebook both land here.
main()
