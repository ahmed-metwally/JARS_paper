# -*- coding: utf-8 -*-
"""
Per-image paired evaluation on the CLCD test split (120 pairs), inference only.

Reuses Ahmed's OWN notebook as the module (no retyped model code), so the
architecture, the dataset reader, the 0.5 threshold and the AMP path are
bit-for-bit the ones that produced the published numbers.

The notebook ends with `if __name__ == "__main__": main()`, so importing it
defines everything WITHOUT training.

Run twice (once per arm), from the same Python that runs the notebooks:

    python perimage_eval.py none
    python perimage_eval.py logit

Writes, per run:   PERIMAGE/perimage_<variant>_seed<seed>.csv
Prints a POOLED check against the published F1. If that check fails, STOP:
the script is not reproducing the original pipeline and nothing downstream
is trustworthy.
"""
import os, sys, json, types, importlib.util
import numpy as np

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
ATT    = os.path.dirname(HERE)                      # ...\attention
os.makedirs(HERE, exist_ok=True)
SRC_NB = os.path.join(ATT, "ABL_CLCD_none_seed_42", "ABL_CLCD_none_seed_42.ipynb")

if not os.path.isfile(SRC_NB):
    raise FileNotFoundError(
        "source notebook not found:\n  %s\nHERE=%r\n"
        "Fix: run this from inside the PERIMAGE folder, use %%run with the full "
        "path, or edit _FALLBACK_DIR at the top of this file." % (SRC_NB, HERE))

# variant -> {seed: (checkpoint dir, published test F1 to verify against)}
RUNS = {
    "none": {
        42:   (os.path.join(ATT, "ABL_CLCD_none_seed_42",   "final"), 0.8076),
        123:  (os.path.join(ATT, "ABL_CLCD_none_seed_123",  "final"), 0.8187),
        2025: (os.path.join(ATT, "ABL_CLCD_none_seed_2025", "final"), 0.8062),
    },
    "logit": {
        42:   (os.path.join(ATT, "BCA_CLCD_518", "CLCD_BALANCEDCA_seed_42_518x518",   "final"), 0.8133),
        123:  (os.path.join(ATT, "BCA_CLCD_518", "CLCD_BALANCEDCA_seed_123_518x518",  "final"), 0.8034),
        2025: (os.path.join(ATT, "BCA_CLCD_518", "CLCD_BALANCEDCA_seed_2025_518x518", "final"), 0.8114),
    },
}


def build_module():
    """Extract the notebook's code cells, redirect OUTPUT_BASE (so the original
    training.log files are never touched), and import the result as a module."""
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")

    needle = 'OUTPUT_BASE = rf"D:\\Master Thesis Codes\\attention\\ABL_{DATASET}_{VARIANT}_seed_{SEED}"'
    if needle not in src:
        raise RuntimeError("OUTPUT_BASE line not found verbatim - the notebook changed; "
                           "fix the patch before running (never let it log into the original folders).")
    src = src.replace(needle, 'OUTPUT_BASE = r"%s"' % os.path.join(HERE, "_scratch").replace("\\", "\\\\"), 1)

    mod_path = os.path.join(HERE, "_cd_module.py")
    with open(mod_path, "w", encoding="utf-8") as f:
        f.write(src)

    spec = importlib.util.spec_from_file_location("_cd_module", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cd_module"] = mod
    spec.loader.exec_module(mod)          # __name__ == "_cd_module" -> main() does NOT run
    return mod


def resolve_test_paths(mod):
    """The two notebooks carry different path strings (the folder was flattened
    between August and October). Accept whichever layout exists on disk now."""
    for root in (r"D:\CLCD\test", r"D:\CLCD\test\test"):
        t1, t2, lb = (os.path.join(root, "time1"),
                      os.path.join(root, "time2"),
                      os.path.join(root, "label"))
        if all(os.path.isdir(p) for p in (t1, t2, lb)):
            return t1, t2, lb
    raise FileNotFoundError("CLCD test split not found under D:\\CLCD\\test or D:\\CLCD\\test\\test")


def per_image_rows(mod, ckpt_dir, variant):
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score

    mod.VARIANT = variant          # read at __init__ and at forward time -> switches the architecture
    t1, t2, lb = resolve_test_paths(mod)
    files1 = mod.load_files(t1, mod.IMG_EXT)
    files2 = mod.load_files(t2, mod.IMG_EXT)
    masks  = mod.load_files(lb, mod.MASK_EXT)
    mod.verify_pairs(files1, files2, masks)
    print("    test pairs: %d" % len(files1))

    ds = mod.ChangeDetectionDataset(files1, files2, masks)
    ldr = mod.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = mod.SiameseViT(dropout=0.26943406970601036).to(mod.DEVICE)
    ck = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)   # strict: any shape/key drift raises
    model.eval()
    n_par = sum(q.numel() for q in model.parameters())
    print("    checkpoint epoch {} | params {:,}".format(ck.get("epoch", "?"), n_par))

    rows, idx = [], 0
    TP = FP = FN = TN = 0
    with torch.no_grad():
        for batch in mod.tqdm(ldr, desc="    %s" % variant):
            i1 = batch["image1"].to(mod.DEVICE, non_blocking=True)
            i2 = batch["image2"].to(mod.DEVICE, non_blocking=True)
            p = mod.predict_probs(model, i1, i2, use_tta=False).cpu().numpy()
            y = mod.to_numpy(batch["mask"])
            for k in range(p.shape[0]):
                prob = p[k].ravel().astype(np.float64)
                gt   = (y[k].ravel() > 0.5).astype(np.int32)
                pr   = (prob >= mod.THRESHOLD).astype(np.int32)
                tp = int(np.sum((pr == 1) & (gt == 1))); fp = int(np.sum((pr == 1) & (gt == 0)))
                fn = int(np.sum((pr == 0) & (gt == 1))); tn = int(np.sum((pr == 0) & (gt == 0)))
                TP += tp; FP += fp; FN += fn; TN += tn
                both = len(np.unique(gt)) >= 2
                rows.append(dict(
                    image_index=idx, file=os.path.basename(files1[idx]),
                    tp=tp, fp=fp, fn=fn, tn=tn,
                    pos_gt=int(gt.sum()), n_pix=int(gt.size), both_classes=int(both),
                    iou=(tp / (tp + fp + fn)) if (tp + fp + fn) else np.nan,
                    f1=(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
                    precision=(tp / (tp + fp)) if (tp + fp) else np.nan,
                    recall=(tp / (tp + fn)) if (tp + fn) else np.nan,
                    auc=(roc_auc_score(gt, prob) if both else np.nan),
                    ap=(average_precision_score(gt, prob) if both else np.nan),
                ))
                idx += 1
    pooled_f1 = 2 * TP / (2 * TP + FP + FN)
    pooled_iou = TP / (TP + FP + FN)
    return rows, pooled_f1, pooled_iou, (TP + FP + FN + TN)


# In a notebook there is no command line: set this and run the cell.
#   "both" = none then logit (what you need anyway) | "none" | "logit"
VARIANT_TO_RUN = "both"


def main(which=None):
    if which is None:
        argv = [a.strip() for a in sys.argv[1:] if a.strip() in RUNS or a.strip() == "both"]
        which = argv[0] if argv else VARIANT_TO_RUN      # argv is unusable inside Jupyter
    variants = list(RUNS) if which == "both" else [which]
    for v in variants:
        if v not in RUNS:
            raise SystemExit("variant must be one of: none, logit, both (got %r)" % v)
    import pandas as pd
    mod = build_module()
    print("module imported OK (no training ran) | arms: %s\n" % ", ".join(variants))
    verdict = []
    for variant in variants:
      for seed, (ckpt_dir, published) in RUNS[variant].items():
        print("=== %s / seed %s ===" % (variant, seed))
        rows, pf1, piou, npix = per_image_rows(mod, ckpt_dir, variant)
        df = pd.DataFrame(rows)
        out = os.path.join(HERE, "perimage_%s_seed%s.csv" % (variant, seed))
        df.to_csv(out, index=False)
        ok = abs(pf1 - published) < 5e-4
        verdict.append((variant, seed, pf1, published, ok))
        print("    pooled F1 = %.4f | published = %.4f | %s" % (pf1, published, "MATCH" if ok else "*** MISMATCH ***"))
        print("    pooled IoU = {:.4f} | pixels = {:,}".format(piou, npix))
        print("    images with both classes: %d / %d" % (int(df.both_classes.sum()), len(df)))
        print("    -> %s\n" % out)
    print("=" * 64)
    for v, s, got, exp, ok in verdict:
        print("  %-6s seed %-5s pooled %.4f  published %.4f  %s" % (v, s, got, exp, "OK" if ok else "MISMATCH"))
    if not all(x[4] for x in verdict):
        print("\nSTOP: a pooled value does not reproduce the published number.")
        print("The per-image files are NOT usable until that is explained.")


if __name__ == "__main__":
    main()
