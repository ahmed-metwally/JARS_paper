# -*- coding: utf-8 -*-
"""
Image-level paired analysis of none vs logit on the CLCD test split.

Reads perimage_<variant>_seed<seed>.csv written by perimage_eval.py.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
It answers: on THIS fixed test set and THESE six trained models, do the two
architectures differ image by image?  n = 120 images instead of 3 seeds.

It does NOT answer: would a fresh retraining of either architecture differ?
That is the seed-level question, and only seed-level replication licenses it.
The image-level p-value is NOT a substitute for the seed-level one; reporting
it as though it were is the error the reviewers rejected the paper for.

The ICC is reported as a DESCRIPTIVE summary only.  It is a one-way ICC on the
images x seeds matrix of delta and its interpretation depends on the variance
model assumed; a low value is not evidence that no architectural effect exists.
The inferential limit stands on its own ground and does not need the ICC: every
delta comes from the same six checkpoints, so a significant p-value here
describes THOSE models on this fixed test set.

(An earlier version of this file converted ICC into an "effective n" with the
cluster design-effect formula n/[1+(m-1)*ICC].  That was wrong: d has already
been averaged over seeds, so each image contributes one observation and the
formula does not apply.  It has been removed, not fixed.)
"""
import os, glob, itertools
import numpy as np, pandas as pd
from scipy import stats

_FALLBACK_DIR = r"D:\Master Thesis Codes\attention\PERIMAGE"

def _here():
    """Works as a .py file AND when the whole file is pasted into a Jupyter cell."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    cwd = os.path.abspath(os.getcwd())
    if os.path.basename(cwd).upper() == "PERIMAGE":
        return cwd
    return _FALLBACK_DIR

HERE   = _here()
SEEDS  = [42, 123, 2025]
METRICS = ["iou", "f1", "auc", "ap"]
TOST_MARGIN_PP = 0.5          # declared equivalence margin, in PERCENTAGE POINTS
BOTH_ONLY = True              # restrict to images with both classes present


def load(variant, seed):
    p = os.path.join(HERE, "perimage_%s_seed%s.csv" % (variant, seed))
    if not os.path.exists(p):
        raise FileNotFoundError("missing %s - run perimage_eval.py %s first" % (p, variant))
    return pd.read_csv(p)


def tost(d, margin):
    """Two one-sided t-tests. Equivalence is claimed only if BOTH reject."""
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); df = n - 1
    t_lo = (m - (-margin)) / se          # H0: mu <= -margin
    t_hi = (m - margin) / se             # H0: mu >= +margin
    p_lo = stats.t.sf(t_lo, df)
    p_hi = stats.t.cdf(t_hi, df)
    return max(p_lo, p_hi), p_lo, p_hi, m, se, df


def icc_oneway(M):
    """ICC(1) from an images x seeds matrix of d values: share of variance
    that is between images rather than between seeds within an image."""
    n, k = M.shape
    gm = M.mean()
    msb = k * ((M.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw = ((M - M.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return icc, msb, msw


def main():
    N = {s: load("none", s) for s in SEEDS}
    L = {s: load("logit", s) for s in SEEDS}

    base = N[SEEDS[0]]
    for s in SEEDS:
        for d in (N[s], L[s]):
            if not (d.image_index.values == base.image_index.values).all():
                raise RuntimeError("image ordering differs between files - cannot pair")

    both = np.ones(len(base), dtype=bool)
    if BOTH_ONLY:
        for s in SEEDS:
            both &= (N[s].both_classes.values == 1) & (L[s].both_classes.values == 1)
    print("images: %d total | %d usable | %d excluded (single-class ground truth)"
          % (len(base), both.sum(), len(base) - both.sum()))
    print("TOST margin: +/-%.2f percentage points (declared in advance)\n" % TOST_MARGIN_PP)

    for met in METRICS:
        M = np.column_stack([(N[s][met].values - L[s][met].values) * 100.0 for s in SEEDS])
        ok = both & np.isfinite(M).all(axis=1)
        M = M[ok]
        d = M.mean(axis=1)                       # average over seeds -> unit = image
        if len(d) < 5:
            print("%-4s : too few usable images (%d)\n" % (met.upper(), len(d))); continue

        w_stat, w_p = stats.wilcoxon(d, alternative="two-sided")
        t_stat, t_p = stats.ttest_1samp(d, 0.0)
        p_tost, p_lo, p_hi, m, se, df = tost(d, TOST_MARGIN_PP)
        ci = m + np.array([-1, 1]) * stats.t.ppf(0.975, df) * se
        icc, msb, msw = icc_oneway(M)
        wins = int((d > 0).sum()); losses = int((d < 0).sum())

        print("=== %s  (none - logit, percentage points) ===" % met.upper())
        print("  n images            : %d" % len(d))
        print("  mean                : %+.4f pp   (95%% CI %+.4f .. %+.4f)" % (m, ci[0], ci[1]))
        print("  median              : %+.4f pp" % np.median(d))
        print("  none better / worse : %d / %d" % (wins, losses))
        print("  Wilcoxon  p         : %.2e" % w_p)
        print("  paired t  p         : %.2e" % t_p)
        print("  TOST +/-%.1fpp      : p = %.4f  ->  %s"
              % (TOST_MARGIN_PP, p_tost,
                 "equivalence SUPPORTED" if p_tost < 0.05 else "equivalence NOT established"))
        print("  ICC(1) of delta     : %.4f   DESCRIPTIVE ONLY" % icc)
        print("     One-way ICC on the images x seeds matrix of delta, before averaging.")
        print("     It is reported as a descriptive summary of how the per-image difference")
        print("     varies across seeds. Its interpretation depends on the variance model")
        print("     assumed, so a low value is NOT evidence that no architectural effect")
        print("     exists, and NOT a decomposition of the residual variance by cause.")
        print("  per-seed, each alone (n=%d images):" % len(d))
        for i, s_ in enumerate(SEEDS):
            col = M[:, i]
            wp = stats.wilcoxon(col).pvalue
            print("    seed %-5s mean %+.3f pp | median %+.3f | none better/worse %d/%d | Wilcoxon p=%.2e"
                  % (s_, col.mean(), np.median(col), int((col > 0).sum()), int((col < 0).sum()), wp))
        agree_n = int((M > 0).all(axis=1).sum()); agree_l = int((M < 0).all(axis=1).sum())
        print("  sign agreement      : none wins in all %d seeds on %d images | logit on %d | mixed %d"
              % (M.shape[1], agree_n, agree_l, len(d) - agree_n - agree_l))
        print()

    print("-" * 68)
    print("Per-seed pooled values, for the record (percentage points):")
    for s in SEEDS:
        for lbl, df_ in (("none", N[s]), ("logit", L[s])):
            tp, fp, fn = df_.tp.sum(), df_.fp.sum(), df_.fn.sum()
            print("  seed %-5s %-6s pooled F1 = %.4f | pooled IoU = %.4f"
                  % (s, lbl, 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)))


if __name__ == "__main__":
    main()
