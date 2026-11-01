# Robust Lithography Hotspot Detection under Lithography-Aware Layout Perturbations

Group 17, AI and Machine Learning for IC (BEVD402L), VIT Chennai.
Kamalesh S (24BVD1086), Hari Ganesh B (24BVD1094), Rupaswathikaa S P (24BVD1095).

This repository contains the code for the DA-3 revision of the project: a lightweight CNN hotspot
detector (12,873 parameters) evaluated on the five ICCAD-12 benchmarks with a held-out validation
set, three training seeds, statistical tests, and two lithography-aware perturbations (CD bias and an
edge-placement-error proxy).

## Files

| File | What it does |
|---|---|
| `DA3_full_pipeline.py` | Builds the train/validation/test split, trains clean and robust models (3 seeds each, 5 benchmarks), evaluates on the test set, and runs the full-test-set perturbation sweep. Writes five CSV files. |
| `DA3_analysis.py` | Reads those CSV files and produces every table, statistical test and figure in the paper. |
| `results/` | The CSV files from our run (add the five files from `/kaggle/working/da3_results/`). |

## Requirements

Python 3 with TensorFlow/Keras, NumPy, pandas, SciPy, scikit-learn and matplotlib (all preinstalled on Kaggle).
A GPU is strongly recommended. Our run used a Kaggle notebook with NVIDIA Tesla T4 GPUs and took 19,030 s (about 5.3 h).

## Dataset

ICCAD-2012 CAD contest hotspot benchmarks, laid out as:

```
<DATA_ROOT>/iccad1/train/train_hs/*.png
<DATA_ROOT>/iccad1/train/train_nhs/*.png
<DATA_ROOT>/iccad1/test/test_hs/*.png
<DATA_ROOT>/iccad1/test/test_nhs/*.png
... (same for iccad2 - iccad5)
```

## How to run

1. In `DA3_full_pipeline.py`, set `DATA_ROOT` to the dataset folder.
2. Run it (on Kaggle: Save Version, then Save & Run All). It writes to `/kaggle/working/da3_results/`:
   - `NEW_table1_train_val_test_counts.csv`: split counts (Table I)
   - `per_seed_test_metrics.csv`: test metrics for every benchmark, model and seed
   - `clean_vs_robust_meanstd.csv`: mean and population std per benchmark and model
   - `clean_vs_robust_significance.csv`: per-benchmark Wilcoxon tests (3 seeds)
   - `full_test_set_sweep_litho_aware.csv`: the perturbation sweep
3. Run the analysis on those files:
   ```
   python DA3_analysis.py results/ analysis_out/
   ```
   This writes `table1.csv` to `table5.csv`, `stats.txt`, `fig1_mean_drop.png` and `fig2_per_benchmark.png`.

## Seeds and settings

| Setting | Value |
|---|---|
| Validation/test split | Official test set split 50/50 within each class, `random_state=123` |
| Training seeds | 0, 1, 2 (Python, NumPy and TensorFlow) |
| Sweep perturbation seed | 999 |
| Epochs | 5 (iccad1-2), 10 (iccad3-5); best epoch chosen by validation balanced accuracy |
| Threshold | 0.5 (fixed) |
| Robust training | each training image perturbed with probability 0.4: CD bias k in {1,2,3} (dilation or erosion) or EPE proxy j in {1,2,3,4} |
| Sweep | CD bias k in {0..4} (dilation; k = 1 is the identity), EPE proxy j in {0..4} px; seed-0 models; full test set |

GPU arithmetic is not bit-deterministic, so a rerun reproduces the numbers closely but not exactly.
Model weights and predicted probabilities are not saved.
