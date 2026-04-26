# MCANet: Multi-label Classification Attention Network for Disaster Damage Recognition

MCANet extends [CSRA (ICCV 2021)](https://arxiv.org/abs/2108.02456) with a Res2Net101 backbone for **multi-label disaster damage classification** on the [RescueNet](https://github.com/BinaLab/RescueNet-Challenge) dataset (10 classes, 4494 aerial images). Paper submitted to JCCE.

## Results (RescueNet test set, 450 images)

| Model | mAP | CF1 | OF1 |
|-------|-----|-----|-----|
| **MCANet (Res2Net101 + CSRA h=1)** | **91.37** | **85.46** | **87.34** |
| Res2Net101 (no attention) | 90.68 | 84.90 | 86.28 |
| ResNet101 (no attention) | 90.50 | 84.48 | 86.45 |
| ViT-B/16-448 | 89.47 | 82.60 | 85.43 |
| MobileNetV2 | 89.54 | 83.84 | 85.52 |
| EfficientNet-B4 | 86.91 | 77.18 | 80.25 |
| VGG-16 | 83.68 | 76.30 | 80.80 |

Multi-head ablation: all h=1..8 outperform the no-attention baseline (90.68%); h=1 achieves highest mAP (91.37).  
Component ablation: class-specific attention (CSRA) beats class-agnostic attention by **+1.52 mAP**.

## Requirements

```
Python 3.8+
PyTorch 1.12.0
CUDA 11.3
torchvision 0.13.0
timm
tqdm
pillow
```

Install:
```bash
pip install -r requirements.txt
```

On PACE HPC:
```bash
module load anaconda3
conda activate csra
```

## Dataset

RescueNet (10 damage classes): Water, Building_No_Damage, Building_Minor_Damage, Building_Major_Damage, Building_Total_Destruction, Vehicle, Road-Clear, Road-Blocked, Tree, Pool.

Raw images should be placed at `Dataset/rescuenet/rescuenet/`. Annotation JSON files are pre-generated in `data/rescuenet/`:

```
data/rescuenet/
  train_rescuenet.json     # 3595 images
  val_rescuenet.json       # 449 images
  test_rescuenet.json      # 450 images (final evaluation only)
  labels/                  # per-image label txt files
```

To regenerate annotations from scratch:
```bash
python utils/prepare/prepare_multilabel_rescuenet.py
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model res2net101_csra --num_heads 1 --lam 0.1 \
    --dataset rescuenet --num_cls 10 \
    --total_epoch 30 --batch_size 16
```

**Supported `--model` values:**
`res2net101_csra`, `res2net101_nocsra`, `res2net101_csra_ca`,
`resnet101_csra`, `resnet101_nocsra`,
`vgg16_nocsra`, `mobilenet_nocsra`, `efficientnet_b4_nocsra`,
`VIT_B16_448_BASE`

**`--num_heads`**: 1–8

**Training config (all experiments):**
- SGD, momentum=0.9, weight_decay=1e-4
- lr=0.01 (backbone), lr×10 (classifier head)
- WarmUpLR (2 epochs) → ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- Seed 42, batch 16, 30 epochs, lam=0.1, threshold=0.5

Checkpoints saved to `checkpoint/{dataset}_{model}_{num_heads}_{timestamp}/epoch_N.pth`.

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model res2net101_csra --num_heads 1 --lam 0.1 \
    --dataset rescuenet --num_cls 10 \
    --load_from checkpoint/rescuenet_res2net101_csra_1_2026-04-03_19-08-48/epoch_30.pth
```

Results are written to `logs/TEST_{dataset}_{model}_{num_heads}_{timestamp}/`.

## Architecture

**CSRA module** (`pipeline/csra.py`): given feature map `[B, C, H, W]`, a 1×1 conv produces class-wise score maps. Each class prediction is `base_logit + λ · att_logit`, where `base_logit` is global average pooled and `att_logit` is the softmax-weighted average at temperature T (T=99 approximates max-pool).

**MHA**: multiple CSRA heads with different temperatures, outputs averaged. Temperature schedule for h heads:
```
h=1: [1]        h=2: [1,99]      h=4: [1,2,4,99]
h=6: [1,2,3,4,5,99]              h=8: [1,2,3,4,5,6,7,99]
```

**MCANet** = Res2Net101 backbone + MHA(CSRA) classification head, trained end-to-end.

## Benchmarking

```bash
CUDA_VISIBLE_DEVICES=0 python profiling/benchmark.py --num_cls 10 --img_size 448 --n_runs 500
```

MCANet (h=1): 43.18M params, 32.56 GFLOPs, 28.82ms latency, 1591.3MB peak memory.

## Attention Visualization

```bash
python attention_visualize/cam_visualize.py
python attention_visualize/cam_multimodel.py
```

Input images: `attention_visualize/input/`, outputs: `attention_visualize/output/`.

## Project Structure

```
CSRA-master/
├── main.py                          # Training entry point
├── test.py                          # Test/evaluation entry point
├── pipeline/                        # Model definitions
│   ├── csra.py                      # CSRA + MHA attention modules
│   ├── res2net101_csra.py           # MCANet (main model)
│   ├── res2net101_nocsra.py         # Res2Net101 baseline
│   ├── resnet{50,101,152}_csra.py   # ResNet variants
│   ├── vit_csra.py                  # ViT-B/16-448
│   └── {vgg16,mobilenet,efficientnet}_nocsra.py
├── res2net/                         # Res2Net source (imported by pipeline)
├── utils/
│   ├── evaluation/                  # AP, mAP, P/R/F1 metrics
│   └── prepare/                     # Annotation generation scripts
├── data/rescuenet/                  # JSON annotations
├── Dataset/rescuenet/               # Raw images
├── figures/                         # Paper figures + generation scripts
├── profiling/                       # Benchmark script + results
├── attention_visualize/             # CAM visualization
├── scripts/                         # Run scripts (*.sh) + SLURM jobs (sbatch/)
├── checkpoint/                      # Saved model checkpoints
├── logs/                            # Per-run evaluation logs
├── run_logs/                        # Historical training run logs
└── reports/                         # Historical experiment reports
```

## Key Result Files

| File | Contents |
|------|----------|
| `experiment_results_v2_final.md` | Raw test numbers for all 16 models + checkpoint paths |
| `experiment_results_full_report.md` | Analysis, benchmark, and paper update summary |
| `experiment_results_paper_narrative.md` | Paper narrative writeup (Apr 22, most recent) |


