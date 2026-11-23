# DGATE: DINOv2 adaptation for multispectral cropland change detection

Code for the paper *DINOv2 adaptation for multispectral cropland change detection: performance, component effects, and acquisition-order sensitivity* by Ahmed Ali Metwally, Amr M. Abdellatif and Mahmoud A. Mahdi (Faculty of Computers and Informatics, Zagazig University, Egypt). Citation details will be added on publication.

DGATE adapts a weight-shared DINOv2-base encoder to four-channel (RGB + NIR) bi-temporal satellite imagery and detects the conversion of cropland to non-agricultural use. Its cross-temporal fusion combines the two directional attention scores before the softmax, so each attention map is row-stochastic by construction, at no parameter cost.

## What this repository contains

Every notebook and script behind a table, figure or number reported in the paper, as run, with its saved outputs. Comments and printed messages, in the code and in the saved outputs, have been translated into English; the code itself is unchanged. The training log of every run is also kept, unedited, in `logs/`. The two `ROWSTRATA` notebooks in `09_attention_row_sums/` were run without saving their outputs. Each training notebook is self-contained: the model, data pipeline, loss, training loop and evaluation are all in the one file, so a single notebook reproduces a single configuration.

## Environment

Python 3.10.19, PyTorch 2.6.0 with CUDA 12.4, `transformers` 5.12.1, NumPy 2.2.6, Albumentations 2.0.8. All runs used a single NVIDIA RTX 4090 with bfloat16 mixed precision.

```
pip install -r requirements.txt
```

The DINOv2-base weights (`facebook/dinov2-base`) are downloaded from Hugging Face on first use.

## Data

The datasets are not redistributed here.

| Dataset | Source | Split used | Layout expected by the notebooks |
|---|---|---|---|
| Hi-CNA (GF-2, RGB + NIR, 0.8 m, 512 × 512) | https://rsidea.whu.edu.cn/Hi-CNA_dataset.htm | official, 4080 / 1358 / 1358 pairs | `train`, `val`, `test`, each with `image1`, `image2` (GeoTIFF) and `change` (PNG; the test masks in `change_255`) |
| CLCD (RGB, 512 × 512) | https://github.com/liumency/CropLand-CD | official, 360 / 120 / 120 pairs | `train`, `val`, `test`, each with `time1`, `time2` and `label` (PNG) |

Dataset and output paths are set near the top of each notebook (`DATA_PATHS`, or the `*_path` variables in the Hi-CNA notebooks, and `OUTPUT_BASE`). They still hold the Windows paths of the original runs; edit them before running.

## From the paper to the code

| Paper | What | Code |
|---|---|---|
| Sec. 3.7 | Hyperparameter search (Optuna, 20 trials at 224 × 224 on Hi-CNA) | `02_hyperparameter_search/optuna_search_224.ipynb` |
| Sec. 3.7 | Scoring at 518 × 518 against the native 512 × 512 | `03_scoring_resolution/` |
| Sec. 3.7 | Parameters, GFLOPs, latency and peak memory | `10_model_complexity/` |
| Sec. 4.1 | Repeat-run difference (two identical seed-2025 CLCD runs) | `01_main_model/clcd_seed2025.ipynb` |
| Sec. 4.2, Tables 1–4 | Main results, three seeds on each benchmark | `01_main_model/` |
| Fig. 2 | Confusion matrix and ROC / PR curves | `11_figures/fig2_confusion_roc_pr.ipynb` |
| Sec. 4.3, Fig. 5 | Acquisition-order reversal on the full Hi-CNA test split and on CLCD | `04_acquisition_order/reversal_*.ipynb` |
| Sec. 4.3.4, Table 5 | Localized order swap (attention blocks vs. fusion head) | `04_acquisition_order/localized_order_swap_hicna.ipynb` |
| Secs. 4.4.1–4.4.3, Table 6, Fig. 6 | Fusion configurations on CLCD; attention-free arm on Hi-CNA | `05_fusion_ablation/` |
| Sec. 4.4.1 | Image-level paired comparison (Wilcoxon, bootstrap) | `06_image_level_paired_test/` |
| Sec. 4.4.4, Tables 7–8 | Row sums of the cross-temporal attention maps | `09_attention_row_sums/` |
| Sec. 4.4.5, Table 9 | Loss ablation | `07_loss_ablation/` |
| Sec. 4.4.6 | Frozen pretrained vs. frozen random encoder | `08_pretrained_vs_random/` |
| All training runs above | Training logs, one per run | `logs/` (index in `logs/README.md`) |

## Running

Open a notebook, set the paths, and run all cells.

- **Seeds.** The seed is the `SEED` variable at the top of each notebook (42, 123 and 2025 in the paper). Where the paper reports several seeds of one configuration and only one notebook is given, the other seeds come from the same notebook with `SEED` changed. The loss-ablation notebooks read the seed and the objective from the environment variables `ABL_SEED` and `ABL_LOSS`.
- **Loss ablation.** `clcd_weighted_bce_only.ipynb` keeps its three code cells as they were run, for seeds 42, 123 and 2025 in that order; the third is a later revision of the same script with the same settings. A stray line holding a single `.`, added to the first two cells after they had run, has been removed so that they run again.
- **Attention row sums.** The notebooks in `09_attention_row_sums/` are generated from the training notebooks by `_mkrowsum.py` and `_mkrowstrata.py`. They run inference only, on the trained checkpoints, and stop if the recomputed attention maps do not reproduce the module's own output.
- **Frozen-encoder comparison.** `make_shared_init.py` writes the single initialization of every non-encoder parameter that the frozen arms share; `finish_accept.py` verifies afterwards, by SHA-256, that the encoder tensors did not change during training. The fine-tuned reference arm is `01_main_model/clcd_seed42.ipynb`.
- **Post-processing.** No reported result uses post-processing: predictions are the sigmoid output thresholded at 0.5. The search notebook still contains CRF and morphological post-processing from an earlier stage; it is applied only to saved visualizations, and the search objective is the plain thresholded validation F1.

## Trained weights

Trained weights are available from the corresponding author, Ahmed Ali Metwally, on reasonable request (a.metwaly021@fci.zu.edu.eg).

## License

Copyright (c) 2026 the authors. All rights reserved; no license is granted for use, copying, modification or redistribution without written permission. See `LICENSE`.
