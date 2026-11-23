########################################################################################################################################################
# BALANCED CROSS-ATTENTION (Fixed Version)
# Mathematical Innovation: Symmetric affinity at the LOGIT level with
# proper row-stochastic normalization for each direction.
########################################################################################################################################################

import os, sys, gc, glob, math, random, warnings
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 42  # single seed is sufficient for a relative ablation comparison

# ══════════════════════════════════════════════════════════════════
#  ATTENTION ABLATION SWITCH - change this, run, repeat.
#    "logit"           : (Ours) symmetrize the affinity at the LOGIT level, before softmax
#    "attention_level" : naive averaging of the two attention maps AFTER softmax
#    "unidirectional"  : a single direction only (reverse is its transpose)
#    "independent"     : two independent softmax directions, no symmetrization
#  Everything else (Q/K/V projections, relative bias, out_proj, loss,
#  threshold, schedule, resolution) is IDENTICAL across variants -> fair ablation.
# ══════════════════════════════════════════════════════════════════
VARIANT = "none"   # logit | attention_level | unidirectional | independent | none | self

# ══════════════════════════════════════════════════════════════════
#  Dataset settings - change only this block to switch between CLCD / MSCC
# ══════════════════════════════════════════════════════════════════
DATASET  = "CLCD"      # label only: "CLCD" or "MSCC"
IMG_EXT  = "png"       # image extension (CLCD/MSCC usually png)
MASK_EXT = "png"       # mask extension
# data paths - adjust to your actual folder structure:
DATA_PATHS = {
    "train_img1": r"D:\CLCD\train\time1", "train_img2": r"D:\CLCD\train\time2", "train_mask": r"D:\CLCD\train\label",
    "val_img1":   r"D:\CLCD\val\time1",   "val_img2":   r"D:\CLCD\val\time2",   "val_mask":   r"D:\CLCD\val\label",
    "test_img1":  r"D:\CLCD\test\time1",  "test_img2":  r"D:\CLCD\test\time2",  "test_mask":  r"D:\CLCD\test\label",
}
# ══════════════════════════════════════════════════════════════════

OUTPUT_BASE = r"D:\\Master Thesis Codes\\attention\\PERIMAGE\\_scratch"
os.makedirs(OUTPUT_BASE, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from scipy.special import expit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import albumentations as A
import rasterio
from rasterio.enums import Resampling
from transformers import Dinov2Model

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    jaccard_score, cohen_kappa_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
    matthews_corrcoef, balanced_accuracy_score, precision_recall_curve
)
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ─── Logging ─────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_BASE, 'training.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────
BATCH_SIZE = 4
ACCUM_STEPS = 16          # res 518: smaller physical batch -> larger accumulation to keep effective ~32
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# full native resolution of CLCD images (512x512): DINOv2 patch=14 does not divide 512 (512/14 not integer).
# 518 = 14x37 is the nearest multiple that fully covers 512 with no pixel loss -> NUM_PATCHES=1369.
IMG_SIZE = 518
PATCH_SIZE = 14
NUM_WORKERS = 0
DROPOUT = 0.1
WEIGHT_DECAY = 0.05
NUM_CHANNELS = 3          # CLCD/MSCC = RGB = 3 channels (Hi-CNA was 4)
MAX_VIS = 30
EMBED_DIM = 768
NUM_HEADS = 12
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
THRESHOLD = 0.5          # single fixed 0.5 threshold for everything: training/monitoring/best-model selection/eval/ablation/ensemble.
TUNE_THRESHOLD = False   # no auto-tuning; everything at 0.5.

assert IMG_SIZE % PATCH_SIZE == 0
logger.info(f"Device: {DEVICE} | AMP: {AMP_DTYPE} | Threshold: {THRESHOLD} (fixed) | "
            f"DINOv2 patch=14 | NUM_PATCHES={NUM_PATCHES} | Channels={NUM_CHANNELS} | "
            f"Seed: {SEED} | ABLATION VARIANT: {VARIANT}")

# ─── Seed ────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seed set to {seed}")

set_seed(SEED)

# ─── GPU Utils ───────────────────────────────────────────
def clear_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

def log_gpu_memory(tag=""):
    if DEVICE.type == "cuda":
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU [{tag}] Alloc:{a:.2f}GB Res:{r:.2f}GB")

# ─── Data I/O ────────────────────────────────────────────
def read_image(image_path):
    """Generic RGB image reader (CLCD/MSCC). Works with 3-channel png/jpg/tif."""
    try:
        im = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        image = np.asarray(im).astype(np.float32) / 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0, 1)
        mean = image.mean(axis=(0, 1), keepdims=True)
        std = image.std(axis=(0, 1), keepdims=True) + 1e-8
        image = (image - mean) / std
        image = np.clip(image, -3, 3)
        image = (image + 3) / 6
        return image
    except Exception as e:
        logger.error(f"Error reading {image_path}: {str(e)}")
        return np.zeros((IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32)

def read_mask_pil(path):
    try:
        m = Image.open(path).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        return (np.array(m) > 0).astype(np.float32)
    except Exception as e:
        logger.error(f"Error reading mask {path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

def verify_pairs(img1, img2, masks=None):
    assert len(img1) == len(img2)
    if masks:
        assert len(img1) == len(masks)
    for i, (f1, f2) in enumerate(zip(img1, img2)):
        b1 = os.path.splitext(os.path.basename(f1))[0]
        b2 = os.path.splitext(os.path.basename(f2))[0]
        assert b1 == b2, f"Mismatch at {i}: {b1} vs {b2}"
    if masks:
        for i, (f1, fm) in enumerate(zip(img1, masks)):
            b1 = os.path.splitext(os.path.basename(f1))[0]
            bm = os.path.splitext(os.path.basename(fm))[0]
            assert b1 == bm, f"Mask mismatch at {i}: {b1} vs {bm}"
    logger.info(f"Pairs verified: {len(img1)} samples")

def load_files(path, ext):
    return sorted(glob.glob(os.path.join(path, f"*.{ext}")))

# ─── Dataset ─────────────────────────────────────────────
class ChangeDetectionDataset(Dataset):
    def __init__(self, img1_files, img2_files, mask_files=None,
                 transform=None, photo_transform=None, is_test=False, cache_size=None):
        self.img1_files = img1_files
        self.img2_files = img2_files
        self.mask_files = mask_files
        self.transform = transform
        self.photo_transform = photo_transform
        self.is_test = is_test
        self.cache = OrderedDict()
        self.cache_size = len(img1_files) // 2 if cache_size is None else cache_size

    def __len__(self):
        return len(self.img1_files)

    def _load(self, path, is_mask=False):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path].copy()
        data = read_mask_pil(path) if is_mask else read_image(path)
        if self.cache_size > 0 and len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[path] = data.copy()
        return data

    def _apply_photo(self, img1, img2):
        """
        Apply identical photometric distortions to both images.
        This prevents pseudo-change artifacts from random augmentation mismatch.
        """
        if self.photo_transform is None:
            return img1, img2
        # Use same random seed for both images to ensure identical transforms
        seed = random.randint(0, 99999)
        random.seed(seed)
        np.random.seed(seed)
        rgb1 = self.photo_transform(image=img1[:, :, :3])['image']
        random.seed(seed)
        np.random.seed(seed)
        rgb2 = self.photo_transform(image=img2[:, :, :3])['image']
        return (
            np.concatenate([rgb1, img1[:, :, 3:]], axis=2).astype(np.float32),
            np.concatenate([rgb2, img2[:, :, 3:]], axis=2).astype(np.float32)
        )

    def __getitem__(self, idx):
        img1 = self._load(self.img1_files[idx])
        img2 = self._load(self.img2_files[idx])
        has_mask = not (self.is_test or self.mask_files is None)
        if has_mask:
            mask = self._load(self.mask_files[idx], is_mask=True)
        if self.transform:
            if has_mask:
                t = self.transform(image=img1, image2=img2, mask=mask)
                img1, img2, mask = t['image'].copy(), t['image2'].copy(), t['mask'].copy()
            else:
                t = self.transform(image=img1, image2=img2)
                img1, img2 = t['image'].copy(), t['image2'].copy()
        img1, img2 = self._apply_photo(img1, img2)
        result = {
            'image1': torch.from_numpy(img1.transpose(2, 0, 1)).float(),
            'image2': torch.from_numpy(img2.transpose(2, 0, 1)).float(),
            'filename': os.path.basename(self.img1_files[idx])
        }
        if has_mask:
            result['mask'] = torch.from_numpy(mask).float()
        return result

# ─── Custom Layers ───────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()

# ─────────────────────────────────────────────────────────────
# BALANCED CROSS-ATTENTION (Mathematically Corrected)
# ─────────────────────────────────────────────────────────────
class BalancedCrossAttention(nn.Module):
    """
    Balanced Cross-Attention with mathematically sound symmetric normalization.

    Mathematical Formulation:
    ─────────────────────────
    Given two token sequences X1 (query) and X2 (key/value), we compute:

        score_12 = (Q1 @ K2^T) / sqrt(d) + bias     [affinity: X1 queries X2]
        score_21 = (Q2 @ K1^T) / sqrt(d) + bias     [affinity: X2 queries X1]

    The innovation is symmetrization at the LOGIT level (before softmax):

        M = (score_12 + score_21^T) / 2

    M[i,j] represents the balanced relational strength between token i in X1
    and token j in X2. By averaging before softmax, both directions share the
    same underlying symmetric affinity structure.

    Proper row-stochastic attention is then obtained via independent softmax:

        S_12 = softmax(M, dim=-1)        [X1 -> X2, rows sum to 1]
        S_21 = softmax(M^T, dim=-1)      [X2 -> X1, rows sum to 1]

    Value aggregation:
        out1 = S_12 @ V2
        out2 = S_21 @ V1

    Properties:
    • Both directions share the same symmetric affinity matrix M.
    • Each attention matrix is properly row-stochastic (valid attention).
    • The positional bias is implicitly symmetrized: (bias[i,j] + bias[j,i])/2.
    • Gradients flow through the shared bottleneck M, enforcing balanced learning.
    """
    def __init__(self, dim, num_heads=NUM_HEADS, dropout=DROPOUT, num_patches=NUM_PATCHES):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        for name in ['q1', 'k1', 'v1', 'q2', 'k2', 'v2']:
            setattr(self, f'{name}_proj', nn.Linear(dim, dim))

        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        gs = int(math.sqrt(num_patches))
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * gs - 1) ** 2, num_heads))
        coords = torch.arange(gs)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), dim=-1).reshape(-1, 2)
        rel_coords = grid[:, None, :] - grid[None, :, :]
        rel_coords[..., 0] += gs - 1
        rel_coords[..., 1] += gs - 1
        rel_coords[..., 0] *= 2 * (gs - 1) + 1
        self.register_buffer('rel_index', rel_coords.sum(-1).long())
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge(self, x):
        return x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], -1)

    def forward(self, x1, x2):
        q1, k1, v1 = self._split(self.q1_proj(x1)), self._split(self.k1_proj(x1)), self._split(self.v1_proj(x1))
        q2, k2, v2 = self._split(self.q2_proj(x2)), self._split(self.k2_proj(x2)), self._split(self.v2_proj(x2))

        N = x1.shape[1]
        bias = self.relative_bias_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1).unsqueeze(0)

        # 1. Directional affinity scores (logits)
        score_12 = (q1 @ k2.transpose(-2, -1)) * self.scale + bias  # X1 queries X2
        score_21 = (q2 @ k1.transpose(-2, -1)) * self.scale + bias  # X2 queries X1
        
        ##############################################################
        # 2. ABLATION SWITCH: where the two directions are combined.
        ##############################################################
        if VARIANT == "logit":
                  ###############
            
            # (Ours) symmetrize at the LOGIT level, before softmax:
            # a shared bottleneck M -> coupled gradients + row-stochastic outputs.
            M = (score_12 + score_21.transpose(-2, -1)) / 2.0
            S_12 = F.softmax(M, dim=-1).type_as(x1)                    # X1 -> X2
            S_21 = F.softmax(M.transpose(-2, -1), dim=-1).type_as(x2)  # X2 -> X1
        elif VARIANT == "attention_level":
            # Naive: average the two attention maps AFTER softmax
            # (breaks row-stochasticity, decouples the gradient paths).
            a12 = F.softmax(score_12, dim=-1).type_as(x1)
            a21 = F.softmax(score_21, dim=-1).type_as(x2)
            S_12 = 0.5 * (a12 + a21.transpose(-2, -1))
            S_21 = 0.5 * (a21 + a12.transpose(-2, -1))
        elif VARIANT == "unidirectional":
            # Single direction only; the reverse is just its transpose
            # -> directional bias, no temporal symmetry.
            S_12 = F.softmax(score_12, dim=-1).type_as(x1)
            S_21 = S_12.transpose(-2, -1)
        elif VARIANT == "independent":
            # Two independent softmax directions, no symmetrization at all.
            S_12 = F.softmax(score_12, dim=-1).type_as(x1)
            S_21 = F.softmax(score_21, dim=-1).type_as(x2)
        elif VARIANT == "self":
            # Parameter-matched control: identical block, identical parameter count,
            # but each branch attends to ITSELF -> no cross-temporal exchange at all.
            # Isolates "temporal interaction" from "extra capacity".
            s11 = F.softmax((q1 @ k1.transpose(-2, -1)) * self.scale + bias, dim=-1).type_as(x1)
            s22 = F.softmax((q2 @ k2.transpose(-2, -1)) * self.scale + bias, dim=-1).type_as(x2)
            out1 = self._merge(s11 @ v1)      # v1, NOT v2
            out2 = self._merge(s22 @ v2)      # v2, NOT v1
            out1 = self.dropout(self.out_norm(self.out_proj(out1)))
            out2 = self.dropout(self.out_norm(self.out_proj(out2)))
            return out1, out2
        else:
            raise ValueError(f"Unknown VARIANT: {VARIANT!r}. Use logit | attention_level | "
                             f"unidirectional | independent | none | self")

        # 4. Value aggregation
        out1 = self._merge(S_12 @ v2)
        out2 = self._merge(S_21 @ v1)

        out1 = self.dropout(self.out_norm(self.out_proj(out1)))
        out2 = self.dropout(self.out_norm(self.out_proj(out2)))

        return out1, out2

# ─────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK (Fixed: separate LayerNorms)
# ─────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=4.0,
                 dropout=DROPOUT, drop_path=0.1, num_patches=NUM_PATCHES):
        super().__init__()
        self.norm1_x1 = nn.LayerNorm(dim)
        self.norm1_x2 = nn.LayerNorm(dim)
        self.attn = BalancedCrossAttention(dim, num_heads, dropout, num_patches)

        # Separate LayerNorms for each branch (prevents cross-contamination)
        self.norm2_x1 = nn.LayerNorm(dim)
        self.norm2_x2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
        )
        self.dp = DropPath(drop_path)

    def forward(self, x1, x2):
        attn_out1, attn_out2 = self.attn(self.norm1_x1(x1), self.norm1_x2(x2))
        x1 = x1 + self.dp(attn_out1)
        x2 = x2 + self.dp(attn_out2)
        x1 = x1 + self.dp(self.mlp(self.norm2_x1(x1)))
        x2 = x2 + self.dp(self.mlp(self.norm2_x2(x2)))
        return x1, x2

# ─── Encoder (DINOv2) ─────────────────────────────────────────────
class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_chans=NUM_CHANNELS):
        super().__init__()
        # disabling mmap at load avoids the Windows 'paging file too small' error (os error 1455)
        # which happens when opening safetensors via memory-mapping. Falls back to full RAM load on older versions.
        try:
            self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", disable_mmap=True)
        except TypeError:
            self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", low_cpu_mem_usage=False)
        # RGB (3 channels) -> keep the original projection. >3 -> adapt it with a mean-filter init for extra channels.
        if in_chans != 3:
            old = self.vit.embeddings.patch_embeddings.projection
            new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride,
                            old.padding, bias=old.bias is not None)
            new.weight.data[:, :3] = old.weight.data
            avg = old.weight.data.mean(dim=1, keepdim=True)
            for c in range(3, in_chans):
                new.weight.data[:, c:c+1] = avg.clone()
            if old.bias is not None:
                new.bias.data = old.bias.data.clone()
            self.vit.embeddings.patch_embeddings.projection = new
            self.vit.config.num_channels = in_chans
            self.vit.embeddings.patch_embeddings.num_channels = in_chans
        for i in range(6):
            for p in self.vit.encoder.layer[i].parameters():
                p.requires_grad = False
        self.embed_dim = self.vit.config.hidden_size
        self.num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

    def forward(self, x):
        out = self.vit(pixel_values=x,
                       output_hidden_states=True,
                       interpolate_pos_encoding=True)
        hs = out.hidden_states
        H = W = IMG_SIZE // PATCH_SIZE
        to_s = lambda h: h[:, 1:].transpose(1, 2).reshape(x.size(0), self.embed_dim, H, W)
        skips = [to_s(hs[i]) for i in [0, 4, 8, 12]]
        return hs[-1][:, 1:], skips

# ─── Siamese Model ───────────────────────────────────────
class SiameseViT(nn.Module):
    def __init__(self, in_chans=NUM_CHANNELS, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, mlp_ratio=4.0, dropout=DROPOUT):
        super().__init__()
        self.encoder = VisionTransformerEncoder(in_chans)
        np_ = self.encoder.num_patches
        # VARIANT == "none"  ->  NO cross-attention at all (DINOv2 + simple fusion baseline).
        # The forward loop below then becomes a no-op: encoder output goes straight
        # to cat([x1, x2, |x1-x2|]) -> fusion -> same skip-connected decoder.
        self.cross_blocks = nn.ModuleList([] if VARIANT == "none" else [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout,
                             drop_path=0.1 * i / max(3, 1), num_patches=np_)
            for i in range(4)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.skip_proj0 = nn.Conv2d(embed_dim * 3, embed_dim * 2, 1)
        self.skip_proj1 = nn.Conv2d(embed_dim * 2 + embed_dim * 2, embed_dim, 1)
        self.skip_proj2 = nn.Conv2d(embed_dim + embed_dim * 2, embed_dim // 2, 1)
        self.skip_proj3 = nn.Conv2d(embed_dim // 2 + embed_dim * 2, embed_dim // 4, 1)

        def dec_block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )

        self.decoder = nn.ModuleList([
            dec_block(embed_dim * 2, embed_dim * 2),
            dec_block(embed_dim, embed_dim),
            dec_block(embed_dim // 2, embed_dim // 2),
            dec_block(embed_dim // 4, embed_dim // 4),
            nn.Sequential(
                nn.Conv2d(embed_dim // 4, embed_dim // 8, 3, padding=1),
                nn.BatchNorm2d(embed_dim // 8), nn.ReLU(inplace=True),
                nn.Dropout2d(dropout), nn.Conv2d(embed_dim // 8, 1, 1)
            )
        ])

    def forward(self, img1, img2):
        B = img1.shape[0]
        x1, feat1 = self.encoder(img1)
        x2, feat2 = self.encoder(img2)

        for block in self.cross_blocks:
            x1, x2 = block(x1, x2)

        diff = torch.abs(x1 - x2)
        x = self.fusion(torch.cat([x1, x2, diff], dim=-1))
        H = W = IMG_SIZE // PATCH_SIZE
        x = x.transpose(1, 2).reshape(B, EMBED_DIM, H, W)
        x = self.skip_proj0(torch.cat([x, torch.cat([feat1[0], feat2[0]], dim=1)], dim=1))
        x = self.decoder[0](x)
        for i in range(1, 4):
            skip = F.interpolate(torch.cat([feat1[i], feat2[i]], dim=1),
                                 size=x.shape[2:], mode='bilinear', align_corners=False)
            x = getattr(self, f'skip_proj{i}')(torch.cat([x, skip], dim=1))
            x = self.decoder[i](x)
        out = self.decoder[4](x)
        out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return out.squeeze(1)

# ─── Loss ────────────────────────────────────────────────
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inp, tgt):
        p = torch.clamp(torch.sigmoid(inp).view(-1), 1e-7, 1 - 1e-7)
        t = tgt.view(-1)
        TP = (p * t).sum()
        FP = ((1 - t) * p).sum()
        FN = (t * (1 - p)).sum()
        ti = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - ti) ** self.gamma

class HybridLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.ft = FocalTverskyLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))

    def update_pos_weight(self, pw):
        self.bce.pos_weight = torch.tensor(pw).to(DEVICE)

    def forward(self, inp, tgt):
        return 0.7 * self.ft(inp, tgt) + 0.3 * self.bce(inp, tgt)

# ─── Helpers ─────────────────────────────────────────────
def to_numpy(t):
    return t.detach().float().cpu().numpy()

def compute_pos_weight(mask_files, n=200):
    pos = neg = 0
    for f in random.sample(mask_files, min(n, len(mask_files))):
        m = read_mask_pil(f)
        pos += m.sum()
        neg += (1 - m).sum()
    pw = float(neg / (pos + 1e-8))
    logger.info(f"pos_weight = {pw:.2f}")
    return pw

def build_scheduler(optimizer, warmup_epochs, total_epochs):
    return sched.SequentialLR(optimizer, [
        sched.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
        sched.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    ], milestones=[warmup_epochs])

def build_scaler():
    return GradScaler(device='cuda') if DEVICE.type == 'cuda' else GradScaler(device='cpu', enabled=False)

def make_loaders(bs, train_ds, val_ds, test_ds):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=None)
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=bs, shuffle=False, **kw),
            DataLoader(test_ds, batch_size=bs, shuffle=False, **kw))

# ─── Metrics ─────────────────────────────────────────────
def compute_metrics(labels, preds, probs, per_image_data=None, use_all_data=False):
    MAX_SAMPLES = 500_000
    if not use_all_data and len(labels) > MAX_SAMPLES:
        idx = np.random.choice(len(labels), MAX_SAMPLES, replace=False)
        labels = labels[idx]
        preds = preds[idx]
        probs = probs[idx]
    probs = np.clip(probs.astype(np.float64), 0.0, 1.0)
    labels = labels.astype(np.int32)
    preds = preds.astype(np.int32)
    m = {
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary', zero_division=0),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall': recall_score(labels, preds, average='binary', zero_division=0),
        'iou': jaccard_score(labels, preds, average='binary', zero_division=0),
        'kappa': cohen_kappa_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds),
    }
    if len(np.unique(labels)) >= 2:
        m['auc'] = roc_auc_score(labels, probs)
        m['ap'] = average_precision_score(labels, probs)
    else:
        m['auc'] = m['ap'] = 0.0
    if per_image_data:
        aucs = [roc_auc_score(l, p) for l, p in per_image_data if len(np.unique(l)) >= 2]
        m['mean_per_image_auc'] = float(np.mean(aucs)) if aucs else 0.0
    return m

# ─── Visualization ───────────────────────────────────────
def _vis_dir(base, epoch):
    d = os.path.join(base, f"epoch_{epoch:03d}", "visualizations")
    os.makedirs(d, exist_ok=True)
    return d

def _plots_dir(base, epoch):
    d = os.path.join(base, f"epoch_{epoch:03d}", "plots")
    os.makedirs(d, exist_ok=True)
    return d

def save_vis(img1, img2, mask, pred, fname, base, epoch):
    def stretch(img):
        lo, hi = np.percentile(img, (2, 98))
        return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    def clahe(u8):
        c = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        return cv2.merge([c.apply(ch) for ch in cv2.split(u8)])
    rgb1 = clahe((stretch(img1[:, :, :3]) * 255).astype(np.uint8))
    rgb2 = clahe((stretch(img2[:, :, :3]) * 255).astype(np.uint8))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, t in zip(ax,
                        [rgb1, rgb2, (mask > 0.5).astype(np.uint8)*255, pred.astype(np.uint8)*255],
                        ["Image 1","Image 2","Ground Truth","Prediction"]):
        a.imshow(im, cmap='gray' if im.ndim == 2 else None)
        a.set_title(t)
        a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(_vis_dir(base, epoch), f"vis_{fname.replace('.tif','.png')}"),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

def save_cm(labels, preds, phase, epoch, base):
    d = _plots_dir(base, epoch)
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(cm[0,0]) if cm.size else 0
        fp = fn = tp = 0
    if phase != "Train":
        sep = "-" * 45
        print(f"\n{sep}\n CM — {phase} | Epoch {epoch+1}\n{sep}")
        print(f" {'':12s} Pred NO Pred YES")
        print(f" {'True NO':12s} {tn:>8d} {fp:>8d} (Spec:{tn/(tn+fp+1e-8):.4f})")
        print(f" {'True YES':12s} {fn:>8d} {tp:>8d} (Rec: {tp/(tp+fn+1e-8):.4f})")
        print(f" TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}\n{sep}\n")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Change","Change"], yticklabels=["No Change","Change"],
                linewidths=0.5, linecolor="gray", cbar_kws={"shrink": 0.8})
    ax.set_title(f"{phase} CM — Epoch {epoch+1}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for (r, c), val in np.ndenumerate(cm):
        lbl = 'TP' if r==1 and c==1 else ('TN' if r==0 and c==0 else ('FP' if r==0 else 'FN'))
        ax.text(c+0.5, r+0.7, f"({lbl})", ha="center", va="center", fontsize=9,
                color="white" if val > cm.max()*0.6 else "black", alpha=0.7)
    plt.tight_layout()
    path = os.path.join(d, f"{phase}_cm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CM saved → {path}")

def save_roc(labels, probs, phase, epoch, base):
    d = _plots_dir(base, epoch)
    if len(np.unique(labels)) < 2:
        return
    try:
        fpr, tpr, thr = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        ap_val = average_precision_score(labels, probs)
        bi = int(np.argmax(tpr - fpr))
        if phase != "Train":
            sep = "-" * 45
            print(f"{sep}\n ROC — {phase} | Epoch {epoch+1}\n{sep}")
            print(f" AUC:{auc_val:.4f} | AP:{ap_val:.4f}")
            print(f" Best Thr:{thr[bi]:.3f} → TPR={tpr[bi]:.4f} FPR={fpr[bi]:.4f}\n{sep}\n")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auc_val:.4f}")
        ax1.scatter(fpr[bi], tpr[bi], color="tomato", zorder=5, s=80, label=f"Thr={thr[bi]:.2f}")
        ax1.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
        ax1.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        ax1.set(title=f"{phase} ROC — E{epoch+1}", xlabel="FPR", ylabel="TPR",
                xlim=[-0.01,1.01], ylim=[-0.01,1.01])
        ax1.legend()
        ax1.grid(alpha=0.3)
        prec, rec, _ = precision_recall_curve(labels, probs)
        ax2.plot(rec, prec, color="darkorange", lw=2, label=f"AP={ap_val:.4f}")
        ax2.fill_between(rec, prec, alpha=0.08, color="darkorange")
        ax2.axhline(labels.sum()/len(labels), color="k", linestyle="--", lw=1, alpha=0.5)
        ax2.set(title=f"{phase} PR — E{epoch+1}", xlabel="Recall", ylabel="Precision",
                xlim=[-0.01,1.01], ylim=[-0.01,1.01])
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(d, f"{phase}_roc_pr.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC+PR saved → {path}")
    except Exception as e:
        logger.warning(f"save_roc failed ({phase} E{epoch+1}): {e}")

def fmt_metrics(phase, loss, m):
    return (f"{phase} - Loss:{loss:.4f} | Acc:{m.get('accuracy',0):.4f} | "
            f"F1:{m.get('f1',0):.4f} | Prec:{m.get('precision',0):.4f} | "
            f"Rec:{m.get('recall',0):.4f} | IoU:{m.get('iou',0):.4f} | "
            f"AUC:{m.get('auc',0):.4f} | AP:{m.get('ap',0):.4f}")

def print_epoch(epoch, t_loss, t_m, v_loss, v_m):
    sep = "=" * 75
    print(f"\n{sep}\n Epoch {epoch+1}/{NUM_EPOCHS}\n{sep}")
    print(fmt_metrics("Train", t_loss, t_m))
    print(fmt_metrics("Val ", v_loss, v_m) + f" | Thr:{THRESHOLD:.2f} (fixed)")
    print("-" * 75)
    for tag, m in [("Train", t_m), ("Val ", v_m)]:
        line = (f" {tag} → Kappa:{m.get('kappa',0):.4f} | MCC:{m.get('mcc',0):.4f} | "
                f"BalAcc:{m.get('balanced_accuracy',0):.4f}")
        if 'mean_per_image_auc' in m:
            line += f" | MeanImgAUC:{m['mean_per_image_auc']:.4f}"
        print(line)
    print(sep)

# ─── EMA (Exponential Moving Average of weights) ─────────
class EMAHelper:
    """Exponential moving average of weights; tracks trainable parameters,
    and automatically handles the ViT parameters unfrozen at epoch 10."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone().float()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone().float()
            else:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ─── train_epoch ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch, initial_pw):
    model.train()
    total_loss = 0.0
    logits_list = []
    labels_list = []
    optimizer.zero_grad()
    criterion.update_pos_weight(initial_pw * max(0.5, 1.0 - epoch * 0.01))
    for i, batch in enumerate(tqdm(loader, desc=f"Train E{epoch+1}")):
        img1 = batch['image1'].to(device, non_blocking=True)
        img2 = batch['image2'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=AMP_DTYPE):
            out = model(img1, img2)
            loss = criterion(out, mask) / ACCUM_STEPS
        loss = torch.nan_to_num(loss, nan=0., posinf=1e3, neginf=-1e3)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf at batch {i}. Skipping.")
            optimizer.zero_grad()
            continue
        scaler.scale(loss).backward()
        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUM_STEPS
        with torch.no_grad():
            lo = to_numpy(out).flatten()
            la = to_numpy(mask).flatten().astype(np.int32)
            if lo.size > 100000:  # subsample to cap train-metric RAM at high resolution (monitoring only)
                idx = np.random.choice(lo.size, 100000, replace=False)
                lo, la = lo[idx], la[idx]
            logits_list.append(lo)
            labels_list.append(la)
    if logits_list:
        all_logits = np.concatenate(logits_list)
        all_labels = np.concatenate(labels_list)
        all_probs = expit(all_logits).astype(np.float64)
        all_preds = (all_probs > THRESHOLD).astype(np.int32)
        metrics = compute_metrics(all_labels, all_preds, all_probs)
    else:
        metrics = {k: 0.0 for k in ['accuracy','balanced_accuracy','f1','precision',
                                    'recall','iou','kappa','mcc','auc','ap']}
    metrics['best_threshold'] = THRESHOLD
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, metrics, all_labels, all_preds, all_probs

# ─── evaluate ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device, epoch,
             phase="Val", base_dir=OUTPUT_BASE, max_vis=MAX_VIS):
    model.eval()
    total_loss = 0.0
    logits_list, labels_list, per_image_data, vis_data = [], [], [], []
    vis_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=phase):
            img1 = batch['image1'].to(device, non_blocking=True)
            img2 = batch['image2'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=AMP_DTYPE):
                out = model(img1, img2)
                loss = criterion(out, mask)

            total_loss += torch.nan_to_num(loss, nan=0., posinf=1e3, neginf=-1e3).item()

            out_np = to_numpy(out)
            mask_np = to_numpy(mask)
            img1_np = to_numpy(batch['image1']).transpose(0, 2, 3, 1)
            img2_np = to_numpy(batch['image2']).transpose(0, 2, 3, 1)

            logits_list.append(out_np.reshape(out_np.shape[0], -1))
            labels_list.append(mask_np.reshape(mask_np.shape[0], -1))

            for i, fname in enumerate(batch['filename']):
                lbl_i = mask_np[i].flatten().astype(np.int32)
                if len(np.unique(lbl_i)) >= 2:
                    per_image_data.append((lbl_i, expit(out_np[i]).flatten()))

                if vis_count < max_vis:
                    vis_data.append((img1_np[i], img2_np[i], mask_np[i], out_np[i].copy(), fname))
                    vis_count += 1

    all_logits = np.concatenate(logits_list, axis=0).flatten()
    all_labels = np.concatenate(labels_list, axis=0).flatten().astype(np.int32)
    all_probs = expit(all_logits.astype(np.float32))          # float32: half the RAM of float64
    all_preds = (all_probs > THRESHOLD).astype(np.int32)

    metrics = compute_metrics(all_labels, all_preds, all_probs, per_image_data)
    metrics['best_threshold'] = THRESHOLD

    for (img1_i, img2_i, mask_i, logits_i, fname) in vis_data:
        pred_vis = (expit(logits_i) > THRESHOLD).astype(np.uint8)
        save_vis(img1_i, img2_i, mask_i, pred_vis, fname, base_dir, epoch)

    return total_loss / max(len(loader), 1), metrics, all_labels, all_preds, all_probs

def run_test(model, loader, device, epoch, base_dir=OUTPUT_BASE):
    model.eval()
    preds_list, labels_list, probs_list, per_image_data = [], [], [], []
    has_mask = False
    pred_dir = os.path.join(base_dir, "predicted_maps")
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            img1 = batch['image1'].to(device, non_blocking=True)
            img2 = batch['image2'].to(device, non_blocking=True)
            has_mask = 'mask' in batch

            with autocast(device_type=device.type, dtype=AMP_DTYPE):
                out = model(img1, img2)

            out_np = to_numpy(out)
            probs_np = expit(out_np)

            img1_np = to_numpy(batch['image1']).transpose(0, 2, 3, 1)
            img2_np = to_numpy(batch['image2']).transpose(0, 2, 3, 1)
            mask_np = to_numpy(batch['mask']) if has_mask else np.zeros_like(probs_np)

            for i, fname in enumerate(batch['filename']):
                pred_i = (probs_np[i] > THRESHOLD).astype(np.uint8)

                cv2.imwrite(os.path.join(pred_dir, f"pred_{fname.replace('.tif','.png')}"),
                            (pred_i * 255).astype(np.uint8))

                save_vis(img1_np[i], img2_np[i], mask_np[i], pred_i, fname, base_dir, epoch)

                if has_mask:
                    lbl_i = mask_np[i].flatten().astype(np.int32)
                    preds_list.append(pred_i.flatten())
                    labels_list.append(lbl_i)
                    probs_list.append(probs_np[i].flatten())
                    if len(np.unique(lbl_i)) >= 2:
                        per_image_data.append((lbl_i, probs_np[i].flatten()))

    if has_mask and preds_list:
        all_labels = np.concatenate(labels_list).astype(np.int32)
        all_probs = np.concatenate(probs_list).astype(np.float64)
        final_preds = (all_probs > THRESHOLD).astype(np.int32)

        metrics = compute_metrics(all_labels, final_preds, all_probs, per_image_data, use_all_data=True)
        metrics['best_threshold'] = THRESHOLD

        logger.info(f"Test → F1:{metrics['f1']:.4f} | IoU:{metrics['iou']:.4f} | AUC:{metrics['auc']:.4f}")
        save_cm(all_labels, final_preds, "Test", epoch, base_dir)
        save_roc(all_labels, all_probs, "Test", epoch, base_dir)
        return metrics, all_labels, final_preds, all_probs

    logger.info("No ground truth for test set.")
    return None, None, None, None

# ─── Checkpoint ──────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best=False, out_dir=os.path.join(OUTPUT_BASE, "final")):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_f1': metrics.get('f1', 0.0),
        'threshold': THRESHOLD,
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
    }
    torch.save(ckpt, os.path.join(out_dir, f'ckpt_epoch_{epoch:03d}.pth'))
    if is_best:
        torch.save(ckpt, os.path.join(out_dir, 'best_model.pth'))
        logger.info(f"Best model saved — F1:{metrics['f1']:.4f} | Thr:{THRESHOLD:.2f} (fixed)")

def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        logger.warning(f"Checkpoint not found: {path}")
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'rng' in ckpt:
        rs = ckpt['rng']
        torch.set_rng_state(rs['torch'])
        if rs.get('cuda') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rs['cuda'])
        np.random.set_state(rs['numpy'])
        random.setstate(rs['python'])
    start = ckpt.get('epoch', 0) + 1
    logger.info(f"Resumed from epoch {start} | F1={ckpt.get('best_f1',0):.4f} | Thr:{THRESHOLD:.2f} (fixed)")
    return start, ckpt.get('best_f1', 0.0)

# ─── Monitor ─────────────────────────────────────────────
class TrainingMonitor:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.no_improve = 0
        self.train_hist = {'loss': [], 'metrics': []}
        self.val_hist = {'loss': [], 'metrics': []}

    def update(self, tl, vl, tm, vm, epoch):
        self.train_hist['loss'].append(tl)
        self.train_hist['metrics'].append(tm)
        self.val_hist['loss'].append(vl)
        self.val_hist['metrics'].append(vm)
        improved = vm['f1'] > self.best_f1 + self.min_delta   # best-model selection by F1 at 0.5
        if improved:
            self.best_f1 = vm['f1']
            self.no_improve = 0
        else:
            self.no_improve += 1
        if abs(tl - vl) > 0.15 or abs(tm['f1'] - vm['f1']) > 0.25:
            logger.warning(f"Overfitting E{epoch+1} | loss_gap={abs(tl-vl):.3f} f1_gap={abs(tm['f1']-vm['f1']):.3f}")
        return improved, self.no_improve >= self.patience

    def plot(self, path=os.path.join(OUTPUT_BASE, 'curves.png')):
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax[0,0].plot(self.train_hist['loss'], label='Train')
        ax[0,0].plot(self.val_hist['loss'], label='Val')
        ax[0,0].set_title('Loss')
        ax[0,0].legend()
        for i, k in enumerate(['f1','iou','accuracy','precision','recall']):
            r, c = (i+1)//3, (i+1)%3
            ax[r,c].plot([m.get(k,0) for m in self.train_hist['metrics']], label='Train')
            ax[r,c].plot([m.get(k,0) for m in self.val_hist['metrics']], label='Val')
            ax[r,c].set_title(k.title())
            ax[r,c].legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

# ─── Summary ─────────────────────────────────────────────
def save_summary(metrics, out_dir=OUTPUT_BASE):
    if not metrics:
        return
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{k: v for k, v in metrics.items() if isinstance(v, float)}]).to_csv(
        os.path.join(out_dir, "test_metrics.csv"), index=False, float_format='%.4f')
    with open(os.path.join(out_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== FINAL TEST RESULTS ===\n\n")
        f.write(f"Threshold: {THRESHOLD:.2f} (fixed)\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:25s}: {v:.4f}\n")
    print("\n" + "="*75 + "\nFINAL TEST RESULTS\n" + "="*75)
    print(f" {'threshold':25s}: {THRESHOLD:.4f} (fixed)")
    for k in ['accuracy','balanced_accuracy','f1','precision','recall','iou','kappa','mcc','auc','ap','mean_per_image_auc']:
        if k in metrics:
            print(f" {k:25s}: {metrics[k]:.4f}")
    print("="*75)

# ─── Main ────────────────────────────────────────────────
def main():
    logger.info("Loading dataset paths...")

    D = DATA_PATHS
    for p in [D["train_img1"], D["train_img2"], D["train_mask"],
              D["val_img1"], D["val_img2"], D["val_mask"],
              D["test_img1"], D["test_img2"]]:
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")

    train_img1 = load_files(D["train_img1"], IMG_EXT)
    train_img2 = load_files(D["train_img2"], IMG_EXT)
    train_masks = load_files(D["train_mask"], MASK_EXT)
    val_img1 = load_files(D["val_img1"], IMG_EXT)
    val_img2 = load_files(D["val_img2"], IMG_EXT)
    val_masks = load_files(D["val_mask"], MASK_EXT)
    test_img1 = load_files(D["test_img1"], IMG_EXT)
    test_img2 = load_files(D["test_img2"], IMG_EXT)
    test_masks = (load_files(D["test_mask"], MASK_EXT)
                  if D["test_mask"] and os.path.exists(D["test_mask"]) else None)

    verify_pairs(train_img1, train_img2, train_masks)
    verify_pairs(val_img1, val_img2, val_masks)
    verify_pairs(test_img1, test_img2, test_masks)

    initial_pos_weight = compute_pos_weight(train_masks)

    geo_tfm = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85,1.15), translate_percent={"x":(-0.05,0.05),"y":(-0.05,0.05)},
                 rotate=(-30,30), shear=(-5,5), p=0.5),
        A.ElasticTransform(alpha=60, sigma=6, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.15),
        A.OpticalDistortion(distort_limit=0.2, p=0.2),
        A.CoarseDropout(num_holes_range=(1,6), hole_height_range=(16,32),
                        hole_width_range=(16,32), fill=0, p=0.2),
    ], additional_targets={'image2':'image','mask':'mask'})

    photo_tfm = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.25),
        A.RandomGamma(gamma_limit=(80,120), p=0.3),
        A.RandomToneCurve(scale=0.1, p=0.2),
        A.OneOf([A.GaussianBlur(blur_limit=(3,7),p=1.), A.MotionBlur(blur_limit=7,p=1.), A.MedianBlur(blur_limit=5,p=1.)], p=0.3),
        A.OneOf([A.GaussNoise(p=1.), A.MultiplicativeNoise(multiplier=(0.9,1.1),p=1.)], p=0.3),
        A.OneOf([A.RandomFog(fog_coef_range=(0.1,0.3),p=1.), A.RandomShadow(shadow_roi=(0,0,1,1),shadow_dimension=5,p=1.)], p=0.15),
    ])

    train_ds = ChangeDetectionDataset(train_img1, train_img2, train_masks, transform=geo_tfm, photo_transform=photo_tfm)
    val_ds = ChangeDetectionDataset(val_img1, val_img2, val_masks)
    test_ds = ChangeDetectionDataset(test_img1, test_img2, test_masks, is_test=(test_masks is None))

    best_lr = 2.8953252238193574e-05
    best_wd = 0.00822526839606462
    best_dr = 0.26943406970601036
    best_bs = 2   # res 518: small physical batch (lower to 1 on OOM and raise ACCUM_STEPS to 32).
                  # effective batch = best_bs x ACCUM_STEPS = 2x16 = 32 (as in the original).
    logger.info(f"Best params: {{'lr': {best_lr}, 'weight_decay': {best_wd}, 'dropout': {best_dr}, 'batch_size': {best_bs}}} | Threshold: {THRESHOLD:.2f} (fixed)")

    t_ldr, v_ldr, ts_ldr = make_loaders(best_bs, train_ds, val_ds, test_ds)

    model = SiameseViT(dropout=best_dr).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd, betas=(0.9,0.999), eps=1e-8)
    criterion = HybridLoss(pos_weight=initial_pos_weight)
    scaler = build_scaler()
    scheduler = build_scheduler(optimizer, 5, NUM_EPOCHS)
    monitor = TrainingMonitor(patience=30)

    final_dir = os.path.join(OUTPUT_BASE, "final")
    os.makedirs(final_dir, exist_ok=True)

    start_epoch = 0
    best_f1 = 0.0
    last_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        last_epoch = epoch
        clear_cache()
        log_gpu_memory(f"Epoch {epoch+1}")
        t_loss, t_m, t_lbl, t_pred, t_prob = train_epoch(model, t_ldr, optimizer, criterion, scaler, DEVICE, epoch, initial_pos_weight)
        v_loss, v_m, v_lbl, v_pred, v_prob = evaluate(model, v_ldr, criterion, DEVICE, epoch, "Val", final_dir)
        print_epoch(epoch, t_loss, t_m, v_loss, v_m)
        save_cm(t_lbl, t_pred, "Train", epoch, final_dir)
        save_roc(t_lbl, t_prob, "Train", epoch, final_dir)
        save_cm(v_lbl, v_pred, "Val", epoch, final_dir)
        save_roc(v_lbl, v_prob, "Val", epoch, final_dir)
        scheduler.step()
        improved, stop = monitor.update(t_loss, v_loss, t_m, v_m, epoch)
        if improved:
            best_f1 = v_m['f1']
            logger.info(f"New best F1:{best_f1:.4f} @ thr={THRESHOLD} | IoU:{v_m['iou']:.4f}")
        save_checkpoint(model, optimizer, scheduler, epoch, v_m, is_best=improved, out_dir=final_dir)

        # ── FIXED: Absolute epoch + add_param_group (preserve optimizer state) ──
        if epoch == 10:  # ABSOLUTE epoch (not relative to start_epoch)
            logger.info(f"Epoch {epoch+1}: Unfreezing ViT encoder...")
            for p in model.encoder.vit.parameters():
                p.requires_grad = True

            # Add new params to existing optimizer (preserves momentum buffers)
            existing_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
            new_vit_params = [p for p in model.encoder.vit.parameters() if p.requires_grad and id(p) not in existing_ids]

            if new_vit_params:
                optimizer.add_param_group({
                    'params': new_vit_params,
                    'lr': 2e-5,
                    'weight_decay': best_wd,
                    'betas': (0.9, 0.999),
                    'eps': 1e-8
                })
                logger.info(f"Added {len(new_vit_params)} ViT parameters to optimizer (preserved momentum)")

            remaining = NUM_EPOCHS - epoch - 1
            if remaining > 0:
                scheduler = sched.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
                logger.info(f"New CosineAnnealingLR: T_max={remaining}, eta_min=1e-6")

        if stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    monitor.plot(os.path.join(final_dir, "training_curves.png"))
    # final evaluation with the best saved model. Threshold fixed at 0.5 by default (fair for ablation).
    ckpt = torch.load(os.path.join(final_dir, 'best_model.pth'), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    thr = find_best_threshold([model], v_ldr, use_tta=False) if TUNE_THRESHOLD else THRESHOLD
    metrics = evaluate_with_threshold([model], ts_ldr, thr, use_tta=False)
    metrics['best_threshold'] = thr
    save_summary(metrics, final_dir)
    logger.info(f"Training complete | test F1:{metrics['f1']:.4f} @ thr={thr:.3f} (tuned={TUNE_THRESHOLD})")

# =====================================================================
#  Inference utilities: TTA + val threshold + Ensemble
# =====================================================================
def _apply_dihedral(x, k, flip):
    if flip:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x

def _invert_dihedral(x, k, flip):
    if k:
        x = torch.rot90(x, -k, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x

@torch.no_grad()
def predict_probs(model, img1, img2, use_tta=False):
    if not use_tta:
        with autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            out = model(img1, img2)
        return torch.sigmoid(out.float())
    probs, n = 0.0, 0
    for k in range(4):
        for flip in (False, True):
            i1 = _apply_dihedral(img1, k, flip)
            i2 = _apply_dihedral(img2, k, flip)
            with autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
                out = model(i1, i2)
            out = _invert_dihedral(out.unsqueeze(1), k, flip).squeeze(1)
            probs = probs + torch.sigmoid(out.float())
            n += 1
    return probs / n

@torch.no_grad()
def collect_ensemble_probs(models, loader, use_tta=False, desc="Eval"):
    labels_list, probs_list, per_image = [], [], []
    for m in models:
        m.eval()
    for batch in tqdm(loader, desc=desc):
        img1 = batch['image1'].to(DEVICE, non_blocking=True)
        img2 = batch['image2'].to(DEVICE, non_blocking=True)
        p = 0.0
        for m in models:
            p = p + predict_probs(m, img1, img2, use_tta=use_tta)
        p = (p / len(models)).cpu().numpy()
        if 'mask' in batch:
            mask = to_numpy(batch['mask'])
            for i in range(p.shape[0]):
                lbl = mask[i].flatten().astype(np.int32)
                pr = p[i].flatten().astype(np.float64)
                labels_list.append(lbl); probs_list.append(pr)
                if len(np.unique(lbl)) >= 2:
                    per_image.append((lbl, pr))
    all_labels = np.concatenate(labels_list).astype(np.int32)
    all_probs = np.concatenate(probs_list).astype(np.float64)
    return all_labels, all_probs, per_image

def find_best_threshold(models, val_loader, use_tta=False, max_pixels=10_000_000, rng_seed=42):
    labels, probs, _ = collect_ensemble_probs(models, val_loader, use_tta=use_tta, desc="Val(thr)")
    if len(labels) > max_pixels:
        rng = np.random.RandomState(rng_seed)
        idx = rng.choice(len(labels), max_pixels, replace=False)
        labels, probs = labels[idx], probs[idx]
    prec, rec, thr = precision_recall_curve(labels, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best = float(thr[np.argmax(f1s[:-1])])
    logger.info(f"Best F1 threshold on VAL = {best:.4f} (val F1={np.max(f1s[:-1]):.4f})")
    return best

def evaluate_with_threshold(models, loader, threshold, use_tta=False):
    labels, probs, per_image = collect_ensemble_probs(models, loader, use_tta=use_tta, desc="Test")
    preds = (probs > threshold).astype(np.int32)
    return compute_metrics(labels, preds, probs, per_image, use_all_data=True)

def _load_model(ckpt_path):
    model = SiameseViT(dropout=0.26943406970601036).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    return model

def _build_val_test_loaders():
    D = DATA_PATHS
    val_ds  = ChangeDetectionDataset(load_files(D["val_img1"], IMG_EXT),  load_files(D["val_img2"], IMG_EXT),  load_files(D["val_mask"], MASK_EXT))
    test_ds = ChangeDetectionDataset(load_files(D["test_img1"], IMG_EXT), load_files(D["test_img2"], IMG_EXT), load_files(D["test_mask"], MASK_EXT))
    _, v_ldr, ts_ldr = make_loaders(4, val_ds, val_ds, test_ds)  # res 518: smaller eval batch (lower to 2/1 on OOM, especially with TTA)
    return v_ldr, ts_ldr

def run_ensemble_eval(ckpt_paths, use_tta=True, tune_threshold=False, out_dir=None):
    """Final evaluation with Ensemble+TTA.
    tune_threshold=False -> fixed THRESHOLD (fair/consistent). True -> tuned on val."""
    out_dir = out_dir or os.path.join(OUTPUT_BASE, "ensemble_eval")
    os.makedirs(out_dir, exist_ok=True)
    models = [_load_model(p) for p in ckpt_paths]
    logger.info(f"Loaded {len(models)} model(s) | TTA={use_tta} | tune_threshold={tune_threshold}")
    v_ldr, ts_ldr = _build_val_test_loaders()
    thr = find_best_threshold(models, v_ldr, use_tta=use_tta) if tune_threshold else THRESHOLD
    metrics = evaluate_with_threshold(models, ts_ldr, thr, use_tta=use_tta)
    metrics['best_threshold'] = thr
    import pandas as pd
    pd.DataFrame([{k: v for k, v in metrics.items() if isinstance(v, float)}]).to_csv(
        os.path.join(out_dir, "ensemble_test_metrics.csv"), index=False, float_format='%.4f')
    with open(os.path.join(out_dir, "ensemble_test_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== ENSEMBLE + TTA + VAL-TUNED THRESHOLD ===\n\n")
        f.write(f"n_models: {len(models)} | TTA: {use_tta} | threshold(val): {thr:.4f}\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:25s}: {v:.4f}\n")
    print("\n" + "="*70 + "\n ENSEMBLE FINAL TEST RESULTS\n" + "="*70)
    print(f" n_models={len(models)} | TTA={use_tta} | threshold(val)={thr:.4f}")
    for k in ['accuracy','balanced_accuracy','f1','precision','recall','iou','kappa','mcc','auc','ap','mean_per_image_auc']:
        if k in metrics:
            print(f" {k:25s}: {metrics[k]:.4f}")
    print("="*70)
    return metrics


# ─── Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
