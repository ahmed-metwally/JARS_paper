# Training logs

One file per training run reported in the paper. Each `training.log` is copied byte for byte from the run folder, so it keeps every attempt that was written to it, including starts that were interrupted and later resumed or restarted. Two runs of the weighted-BCE loss ablation wrote no log file; their log lines are taken from the standard output saved in the notebook cell that ran them.

Test F1 is printed in the log only where the notebook evaluated the test split at the end of training; the Hi-CNA test scores were computed afterwards from the saved best-validation checkpoints.

| Log | Notebook | Completed runs in the file | Source |
|---|---|---|---|
| `hicna_seed42.log` | `01_main_model/hicna_seed42.ipynb` | seed 42, to epoch 50, best val F1 0.8395 | copied unchanged |
| `hicna_seed123.log` | `01_main_model/hicna_seed123.ipynb` | seed 123, to epoch 50, best val F1 0.8451 | copied unchanged |
| `hicna_seed2025.log` | `01_main_model/hicna_seed2025.ipynb` | seed 2025: the early-stopping rule stopped training at epoch 23 (best val F1 0.8312); training was resumed from the epoch-23 checkpoint and continued to epoch 47, where writing a checkpoint failed. Best val F1 0.8473 (epoch 44); test F1 0.8535 for that checkpoint, evaluated afterwards. | copied unchanged |
| `clcd_seed42.log` | `01_main_model/clcd_seed42.ipynb` | seed 42, 224 x 224 (earlier, reduced resolution), to epoch 50, best val F1 0.7936, test F1 0.7725; seed 42, to epoch 50, best val F1 0.8343, test F1 0.8133 | copied unchanged |
| `clcd_seed123.log` | `01_main_model/clcd_seed123.ipynb` | seed 123, to epoch 50, best val F1 0.8240, test F1 0.8034 | copied unchanged |
| `clcd_seed2025.log` | `01_main_model/clcd_seed2025.ipynb` | seed 2025, to epoch 50, best val F1 0.8339, test F1 0.8020; seed 2025, to epoch 50, best val F1 0.8252, test F1 0.8114 | copied unchanged |
| `clcd_no_cross_attention_seed42.log` | `05_fusion_ablation/clcd_no_cross_attention.ipynb` | seed 42, to epoch 50, best val F1 0.8182, test F1 0.8076 | copied unchanged |
| `clcd_no_cross_attention_seed123.log` | `05_fusion_ablation/clcd_no_cross_attention.ipynb` | seed 123, to epoch 50, best val F1 0.8157, test F1 0.8187 | copied unchanged |
| `clcd_no_cross_attention_seed2025.log` | `05_fusion_ablation/clcd_no_cross_attention.ipynb` | seed 2025, to epoch 50, best val F1 0.8202, test F1 0.8062 | copied unchanged |
| `clcd_self_attention_control_seed42.log` | `05_fusion_ablation/clcd_self_attention_control_seed42.ipynb` | seed 42, to epoch 50, best val F1 0.8216, test F1 0.8136 | copied unchanged |
| `clcd_self_attention_control_seed123.log` | `05_fusion_ablation/clcd_self_attention_control_seed123.ipynb` | seed 123, to epoch 50, best val F1 0.8247, test F1 0.8032 | copied unchanged |
| `clcd_unidirectional_seed42.log` | `05_fusion_ablation/clcd_unidirectional.ipynb` | seed 42, to epoch 50, best val F1 0.8311, test F1 0.8059 | copied unchanged |
| `clcd_independent_seed42.log` | `05_fusion_ablation/clcd_independent.ipynb` | seed 42, to epoch 50, best val F1 0.8239, test F1 0.8079 | copied unchanged |
| `clcd_attention_level_seed42.log` | `05_fusion_ablation/clcd_attention_level.ipynb` | seed 42, to epoch 50, best val F1 0.8246, test F1 0.8157 | copied unchanged |
| `hicna_no_cross_attention_seed42.log` | `05_fusion_ablation/hicna_no_cross_attention_train.ipynb` | seed 42, to epoch 50, best val F1 0.8447 | copied unchanged |
| `clcd_focal_tversky_only_seed42.log` | `07_loss_ablation/clcd_focal_tversky_only.ipynb` | seed 42, focal_tversky, to epoch 50, best val F1 0.8304, test F1 0.8108 | copied unchanged |
| `clcd_focal_tversky_only_seed123_and_seed2025.log` | `07_loss_ablation/clcd_focal_tversky_only.ipynb` | seed 123, focal_tversky, to epoch 50, best val F1 0.8168, test F1 0.8073; seed 2025, focal_tversky, to epoch 50, best val F1 0.8232, test F1 0.8097 | copied unchanged |
| `clcd_weighted_bce_only_seed2025.log` | `07_loss_ablation/clcd_weighted_bce_only.ipynb` | seed 2025, bce, to epoch 50, best val F1 0.8161, test F1 0.8063 | copied unchanged |
| `clcd_arm_C_pretrained_frozen_seed42.log` | `08_pretrained_vs_random/arm_C_pretrained_frozen_seed42.ipynb` | seed 42, to epoch 50, best val F1 0.7808, test F1 0.7578 | copied unchanged |
| `clcd_arm_B_random_frozen_seed42.log` | `08_pretrained_vs_random/arm_B_random_frozen_seed42.ipynb` | seed 42, to epoch 50, best val F1 0.4619, test F1 0.4748 | copied unchanged |
| `clcd_random_fully_trainable_seed42.log` | `08_pretrained_vs_random/random_fully_trainable_seed42.ipynb` | seed 42, to epoch 50, best val F1 0.5410, test F1 0.5277 | copied unchanged |
| `clcd_weighted_bce_only_seed42.log` | `07_loss_ablation/clcd_weighted_bce_only.ipynb` | seed 42, bce, to epoch 50, best val F1 0.8239, test F1 0.8088 | stdout of the notebook cell |
| `clcd_weighted_bce_only_seed123.log` | `07_loss_ablation/clcd_weighted_bce_only.ipynb` | seed 123, bce, to epoch 50, best val F1 0.8168, test F1 0.7885 | stdout of the notebook cell |
