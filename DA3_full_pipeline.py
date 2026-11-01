"""
DA-3: Robust Lithography Hotspot Detection — REVISED PIPELINE
Group 17 — Rupaswathikaa S P (24BVD1095)

Implements every G17 + common feedback item:
  - Real train/val/test split (val != test; test touched once, at the end)
  - 3-seed training, mean +/- std, paired significance test
  - Lithography-aware perturbations: CD bias (dilation/erosion), EPE (edge jitter)
  - Full-test-set sweep, clean-trained vs robust-trained
  - Robust-training ablation (clean vs robust, same architecture/split)

HOW TO USE (Kaggle):
1. Keep DATA_ROOT pointing at the ICCAD-12 folder (same as DA-2).
2. Run top to bottom. Takes longer than DA-2 (3 seeds x 2 models x 5 benchmarks
   for training; full-set sweep only for 1 seed each to stay feasible).
3. Outputs land in /kaggle/working/da3_results/ : per-split counts (-> new
   Table I), per-seed metrics, significance test results, full sweep CSVs.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import ndimage
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "/kaggle/input/datasets/rupaswathikaasp/iccad-12hotspot-dataset/iccad-official"
BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PER_BENCHMARK = {"iccad1": 5, "iccad2": 5, "iccad3": 10, "iccad4": 10, "iccad5": 10}
SEEDS = [0, 1, 2]
PRIMARY_SEED = SEEDS[0]          # used for the full-test-set perturbation sweep
VAL_FRACTION = 0.5               # of the OLD test set -> split into val/test
RESULTS_DIR = "/kaggle/working/da3_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Perturbation severities (kept comparable to DA-2's 5-level sweep)
EPE_JITTER_PX = [0, 1, 2, 3, 4]          # local per-edge jitter, pixels
CD_BIAS_PX = [0, 1, 2, 3, 4]             # dilation(+)/erosion(-) structuring size

# Augmentation probability used during "robust" training (corrected, not aggressive)
ROBUST_AUG_PROB = 0.40


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Model (unchanged baseline CNN, 12,873 params)
# ---------------------------------------------------------------------------
def build_cnn():
    m = models.Sequential(name="baseline_lightweight_cnn")
    m.add(layers.Conv2D(12, (3, 3), activation='elu', input_shape=(*IMG_SIZE, 3)))
    m.add(layers.Conv2D(12, (3, 3), activation='elu'))
    m.add(layers.Conv2D(12, (3, 3), activation=None))
    m.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    m.add(layers.Activation('elu'))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.MaxPooling2D((5, 5)))
    m.add(layers.Conv2D(12, (3, 3), activation='elu'))
    m.add(layers.Conv2D(12, (3, 3), activation='elu'))
    m.add(layers.Conv2D(12, (3, 3), activation=None))
    m.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    m.add(layers.Activation('elu'))
    m.add(layers.MaxPooling2D((2, 2)))
    m.add(layers.Flatten())
    m.add(layers.Dropout(0.3))
    m.add(layers.Dense(10, activation='relu'))
    m.add(layers.Dense(1, activation='sigmoid'))
    return m


# ---------------------------------------------------------------------------
# NEW: real train/val/test split, built once per benchmark, reused by all seeds
# ---------------------------------------------------------------------------
def list_image_paths(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))])


def build_splits(benchmark):
    """Old DA-2 layout: train/{train_hs,train_nhs}, test/{test_hs,test_nhs}.
    We KEEP the original train set as-is (already genuinely held out from test),
    and split the OLD test set in half, stratified by class, into a real
    validation set and a real (smaller but still fully labeled) test set.
    This is the minimum change that removes "test used as validation" while
    not requiring new data."""
    train_dir = os.path.join(DATA_ROOT, benchmark, "train")
    test_dir = os.path.join(DATA_ROOT, benchmark, "test")

    train_hs = list_image_paths(os.path.join(train_dir, "train_hs"))
    train_nhs = list_image_paths(os.path.join(train_dir, "train_nhs"))
    old_test_hs = list_image_paths(os.path.join(test_dir, "test_hs"))
    old_test_nhs = list_image_paths(os.path.join(test_dir, "test_nhs"))

    def split_half(paths, seed=123):
        if len(paths) < 2:
            return paths, paths  # degenerate guard
        a, b = train_test_split(paths, test_size=VAL_FRACTION,
                                 random_state=seed, shuffle=True)
        return a, b

    val_hs, test_hs = split_half(old_test_hs)
    val_nhs, test_nhs = split_half(old_test_nhs)

    splits = {
        "train": {"hs": train_hs, "nhs": train_nhs},
        "val":   {"hs": val_hs,   "nhs": val_nhs},
        "test":  {"hs": test_hs,  "nhs": test_nhs},
    }
    counts = {f"{s}_{c}": len(splits[s][c]) for s in splits for c in ("hs", "nhs")}
    counts["benchmark"] = benchmark
    return splits, counts


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_dataset(split_dict, batch_size=BATCH_SIZE, shuffle=False, augment_fn=None):
    paths = split_dict["hs"] + split_dict["nhs"]
    labels = [1] * len(split_dict["hs"]) + [0] * len(split_dict["nhs"])
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 1), reshuffle_each_iteration=True)

    def _load(p, y):
        img = load_image(p)
        return img, tf.cast(y, tf.float32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, np.array(labels)


# ---------------------------------------------------------------------------
# NEW: lithography-aware perturbations (replace DA-2's generic translation/
# rotation/scaling/noise with CD bias + EPE, as required by G17 feedback)
# ---------------------------------------------------------------------------
def apply_cd_bias_np(img, bias_px):
    """Critical-Dimension bias: uniformly grows (dilation, bias_px>0) or
    shrinks (erosion, bias_px<0) pattern features, simulating over/under
    exposure. Applied per-channel on the binarized-ish layout image."""
    if bias_px == 0:
        return img
    size = abs(bias_px)
    struct = np.ones((size, size, 1))
    if bias_px > 0:
        return np.clip(ndimage.grey_dilation(img, footprint=struct), 0, 1)
    else:
        return np.clip(ndimage.grey_erosion(img, footprint=struct), 0, 1)


def apply_epe_np(img, jitter_px, rng):
    """Edge Placement Error: instead of a single rigid whole-image shift
    (DA-2's old 'translation'), warp a random local region of the image by a
    small independent offset, approximating local edge jitter at pattern
    boundaries."""
    if jitter_px == 0:
        return img
    h, w, c = img.shape
    out = img.copy()
    # pick 1-3 local patches and jitter each independently
    n_patches = rng.randint(1, 4)
    for _ in range(n_patches):
        ph, pw = rng.randint(h // 4, h // 2), rng.randint(w // 4, w // 2)
        y0 = rng.randint(0, max(h - ph, 1))
        x0 = rng.randint(0, max(w - pw, 1))
        dy = rng.randint(-jitter_px, jitter_px + 1)
        dx = rng.randint(-jitter_px, jitter_px + 1)
        patch = img[y0:y0 + ph, x0:x0 + pw, :]
        shifted = ndimage.shift(patch, shift=(dy, dx, 0), order=0, mode='nearest')
        out[y0:y0 + ph, x0:x0 + pw, :] = shifted
    return out


def make_robust_train_augment(prob=ROBUST_AUG_PROB, seed=0):
    """tf.data-compatible augmentation wrapper: with `prob` chance, apply a
    randomly-chosen CD-bias or EPE perturbation at a random severity. Uses
    tf.numpy_function since ndimage ops aren't natively TF ops."""
    py_rng = random.Random(seed)

    def _augment_np(img):
        if py_rng.random() < prob:
            if py_rng.random() < 0.5:
                bias = py_rng.choice([-3, -2, -1, 1, 2, 3])
                img = apply_cd_bias_np(img, bias)
            else:
                jitter = py_rng.choice([1, 2, 3, 4])
                img = apply_epe_np(img, jitter, py_rng)
        return img.astype(np.float32)

    def _tf_wrap(img, label):
        img2 = tf.numpy_function(_augment_np, [img], tf.float32)
        img2.set_shape([*IMG_SIZE, 3])
        return img2, label

    return _tf_wrap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    bal_acc = (sens + spec) / 2
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": sens, "Specificity": spec,
            "F1": f1, "BalancedAccuracy": bal_acc}


def predict_on(model, ds):
    y_prob = model.predict(ds, verbose=0).ravel()
    return (y_prob >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Stage 1: build splits once, save counts -> new Table I
# ---------------------------------------------------------------------------
def stage1_build_all_splits():
    all_counts = []
    split_cache = {}
    for bm in BENCHMARKS:
        splits, counts = build_splits(bm)
        split_cache[bm] = splits
        all_counts.append(counts)
        print(f"[{bm}] train HS/NHS={counts['train_hs']}/{counts['train_nhs']} "
              f"| val HS/NHS={counts['val_hs']}/{counts['val_nhs']} "
              f"| test HS/NHS={counts['test_hs']}/{counts['test_nhs']}")
    df = pd.DataFrame(all_counts)[
        ["benchmark", "train_hs", "train_nhs", "val_hs", "val_nhs", "test_hs", "test_nhs"]]
    df.to_csv(os.path.join(RESULTS_DIR, "NEW_table1_train_val_test_counts.csv"), index=False)
    print("\nSaved NEW_table1_train_val_test_counts.csv -- this replaces old Table I.")
    return split_cache


# ---------------------------------------------------------------------------
# Stage 2: 3-seed training (clean + robust), val-only model selection
# ---------------------------------------------------------------------------
def train_one(benchmark, splits, seed, robust):
    set_global_seed(seed)
    epochs = EPOCHS_PER_BENCHMARK[benchmark]

    augment_fn = make_robust_train_augment(seed=seed) if robust else None
    train_ds, _ = make_dataset(splits["train"], shuffle=True, augment_fn=augment_fn)
    val_ds, val_labels = make_dataset(splits["val"], shuffle=False)

    model = build_cnn()
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    best_val_bal_acc = -1
    best_weights = None
    for ep in range(epochs):
        model.fit(train_ds, epochs=1, verbose=0)
        y_pred_val = predict_on(model, val_ds)
        m = compute_metrics(val_labels, y_pred_val)
        if m["BalancedAccuracy"] >= best_val_bal_acc:
            best_val_bal_acc = m["BalancedAccuracy"]
            best_weights = model.get_weights()

    model.set_weights(best_weights)   # model selection on VAL only
    return model, best_val_bal_acc


def stage2_train_all_seeds(split_cache):
    """Returns: results_rows (per seed/benchmark/model-type TEST metrics,
    touched exactly once here) and a dict of trained models for the primary
    seed only (clean_primary[bm], robust_primary[bm]) for the Stage-3 sweep."""
    rows = []
    primary_models = {"clean": {}, "robust": {}}
    per_seed_csv = os.path.join(RESULTS_DIR, "per_seed_test_metrics.csv")

    for bm in BENCHMARKS:
        t0 = time.time()
        splits = split_cache[bm]
        test_ds, test_labels = make_dataset(splits["test"], shuffle=False)

        for robust in (False, True):
            tag = "robust" if robust else "clean"
            seed_bal_accs = []
            for seed in SEEDS:
                model, val_bal_acc = train_one(bm, splits, seed, robust)
                y_pred_test = predict_on(model, test_ds)   # test touched ONCE, here
                m = compute_metrics(test_labels, y_pred_test)
                m.update({"benchmark": bm, "model": tag, "seed": seed,
                          "val_BalancedAccuracy": val_bal_acc})
                rows.append(m)
                seed_bal_accs.append(m["BalancedAccuracy"])
                print(f"[{bm}][{tag}][seed={seed}] test BalAcc={m['BalancedAccuracy']:.4f}")

                if seed == PRIMARY_SEED:
                    primary_models[tag][bm] = model
                else:
                    del model
                    tf.keras.backend.clear_session()

        pd.DataFrame(rows).to_csv(per_seed_csv, index=False)   # checkpoint every benchmark
        print(f"[{bm}] Stage-2 training done in {time.time() - t0:.1f}s "
              f"(checkpoint saved: per_seed_test_metrics.csv)")

    df = pd.DataFrame(rows)
    df.to_csv(per_seed_csv, index=False)

    # mean +/- std per benchmark x model, plus paired significance test
    summary_rows = []
    sig_rows = []
    for bm in BENCHMARKS:
        sub = df[df.benchmark == bm]
        clean = sub[sub.model == "clean"].sort_values("seed")["BalancedAccuracy"].values
        robust = sub[sub.model == "robust"].sort_values("seed")["BalancedAccuracy"].values
        for tag, vals in [("clean", clean), ("robust", robust)]:
            summary_rows.append({
                "benchmark": bm, "model": tag,
                "BalancedAccuracy_mean": np.mean(vals), "BalancedAccuracy_std": np.std(vals)
            })
        try:
            stat, p = wilcoxon(clean, robust)
        except ValueError:
            p = float("nan")
        sig_rows.append({"benchmark": bm, "test": "wilcoxon_clean_vs_robust",
                          "n_seeds": len(SEEDS), "p_value": p,
                          "significant_at_0.05": (p < 0.05) if p == p else False})

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "clean_vs_robust_meanstd.csv"), index=False)
    pd.DataFrame(sig_rows).to_csv(
        os.path.join(RESULTS_DIR, "clean_vs_robust_significance.csv"), index=False)
    print("\nSaved clean_vs_robust_meanstd.csv + clean_vs_robust_significance.csv "
          "(robust-training ablation).")

    return primary_models, split_cache


# ---------------------------------------------------------------------------
# Stage 3: full-test-set sweep (CD bias + EPE), clean vs robust, primary seed
# ---------------------------------------------------------------------------
def apply_perturbation_batch(images_np, kind, severity, rng):
    out = np.empty_like(images_np)
    for i in range(images_np.shape[0]):
        if kind == "cd_bias":
            out[i] = apply_cd_bias_np(images_np[i], severity)
        elif kind == "epe":
            out[i] = apply_epe_np(images_np[i], severity, rng)
    return out


SWEEP_CHUNK_SIZE = 256  # images held in memory at once during Stage 3 (keeps RAM bounded)
SWEEP_CSV_PATH = os.path.join(RESULTS_DIR, "full_test_set_sweep_litho_aware.csv")


def _save_sweep_checkpoint(rows):
    """Overwrite the sweep CSV with everything completed so far. Cheap (small
    file) and means a crash on a later benchmark does NOT lose earlier ones."""
    pd.DataFrame(rows).to_csv(SWEEP_CSV_PATH, index=False)


def stage3_full_sweep(primary_models, split_cache):
    sweep_rows = []
    rng = random.Random(999)

    # combos to evaluate per image-chunk: (tag, kind, severity) -- decode each
    # chunk of real images ONCE, then reuse it for every combo, instead of
    # re-reading the whole test set from disk per combo (was ~20x slower)
    combos = []
    for kind, severities in [("cd_bias", CD_BIAS_PX), ("epe", EPE_JITTER_PX)]:
        for sev in severities:
            combos.append((kind, sev))

    for bm in BENCHMARKS:
        t0 = time.time()
        splits = split_cache[bm]
        test_paths = splits["test"]["hs"] + splits["test"]["nhs"]
        test_labels_full = np.array([1] * len(splits["test"]["hs"]) +
                                     [0] * len(splits["test"]["nhs"]))
        n_test = len(test_paths)
        models_by_tag = {"clean": primary_models["clean"][bm],
                          "robust": primary_models["robust"][bm]}

        # accumulator: y_pred_accum[tag][(kind, sev)] = full-length int array
        y_pred_accum = {tag: {c: np.empty(n_test, dtype=int) for c in combos}
                         for tag in models_by_tag}

        for start in range(0, n_test, SWEEP_CHUNK_SIZE):
            end = min(start + SWEEP_CHUNK_SIZE, n_test)
            chunk_paths = test_paths[start:end]
            # decode this chunk ONCE (parallelized), reused for every combo below
            path_ds = tf.data.Dataset.from_tensor_slices(chunk_paths)
            path_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
            chunk_imgs = np.stack(list(path_ds.as_numpy_iterator()), axis=0)

            for tag, model in models_by_tag.items():
                for (kind, sev) in combos:
                    perturbed = apply_perturbation_batch(chunk_imgs, kind, sev, rng)
                    y_prob = model.predict(perturbed, batch_size=BATCH_SIZE,
                                            verbose=0).ravel()
                    y_pred_accum[tag][(kind, sev)][start:end] = (y_prob >= 0.5).astype(int)
                    del perturbed
            del chunk_imgs

        for tag in models_by_tag:
            for (kind, sev) in combos:
                m = compute_metrics(test_labels_full, y_pred_accum[tag][(kind, sev)])
                m.update({"benchmark": bm, "model": tag,
                          "perturbation": kind, "severity": sev,
                          "n_test_full": n_test})
                sweep_rows.append(m)

        elapsed = time.time() - t0
        print(f"[{bm}] Stage-3 sweep done in {elapsed:.1f}s "
              f"({n_test} test images x {len(combos)} combos x 2 models)")
        _save_sweep_checkpoint(sweep_rows)   # checkpoint after EVERY benchmark

    print("\nSaved full_test_set_sweep_litho_aware.csv "
          "(full test set, both models, CD bias + EPE).")
    return pd.DataFrame(sweep_rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("STAGE 1: building real train/val/test splits ...")
    split_cache = stage1_build_all_splits()

    print("\nSTAGE 2: 3-seed training, clean + robust, val-only model selection, "
          "test touched once ...")
    primary_models, split_cache = stage2_train_all_seeds(split_cache)

    print("\nSTAGE 3: full-test-set lithography-aware perturbation sweep "
          "(primary seed only, both models) ...")
    sweep_df = stage3_full_sweep(primary_models, split_cache)

    print("\nALL DONE. Check:", RESULTS_DIR)
