"""
DA-3 analysis: turns the CSVs written by DA3_full_pipeline.py into every table,
statistical test and figure reported in the paper.
Group 17 - Robust Lithography Hotspot Detection under Lithography-Aware Layout Perturbations

Usage:
    python DA3_analysis.py <results_dir> [<out_dir>]
<results_dir> must contain (as written by DA3_full_pipeline.py):
    NEW_table1_train_val_test_counts.csv
    per_seed_test_metrics.csv
    full_test_set_sweep_litho_aware.csv
Outputs (in <out_dir>, default ./analysis_out): table1.csv ... table5.csv,
stats.txt, fig1_mean_drop.png, fig2_per_benchmark.png

Notes
- Standard deviations are sample standard deviations (n - 1).
- Robust-minus-clean differences are rounded to 1e-10 before the Wilcoxon test so
  that differences of equal magnitude are treated as ties (two iccad5 seeds differ
  by exactly +/- one hotspot). With ties SciPy uses the tie-corrected normal
  approximation.
- CD-bias kernel k = 1 is a 1 x 1 structuring element (identity); comparisons over
  CD severities use k = 2, 3, 4.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
# Published per-benchmark results of Yang et al., J. Micro/Nanolith. MEMS MOEMS 16(3) 033504, 2017
# (accuracy = detected hotspots / all hotspots, %, and false-alarm count on the full official test set)
PUBLISHED = {"iccad1": (100.0, 1037), "iccad2": (98.7, 83), "iccad3": (98.0, 3108),
             "iccad4": (94.5, 296), "iccad5": (95.1, 394)}


def ms(x, p=3):
    return f"{np.mean(x):.{p}f} ± {np.std(x, ddof=1):.{p}f}"


def main(rdir, odir):
    os.makedirs(odir, exist_ok=True)
    t1 = pd.read_csv(os.path.join(rdir, "NEW_table1_train_val_test_counts.csv"))
    ps = pd.read_csv(os.path.join(rdir, "per_seed_test_metrics.csv"))
    sw = pd.read_csv(os.path.join(rdir, "full_test_set_sweep_litho_aware.csv"))
    out = []

    # consistency checks
    for r in ps.itertuples():
        row = t1[t1.benchmark == r.benchmark].iloc[0]
        assert r.TP + r.FN == row.test_hs and r.TN + r.FP == row.test_nhs, r
    for (b, m), g in sw.groupby(["benchmark", "model"]):
        s0 = g[g.severity == 0].BalancedAccuracy.values
        seed0 = ps[(ps.benchmark == b) & (ps.model == m) & (ps.seed == 0)].BalancedAccuracy.values[0]
        assert np.allclose(s0, seed0), (b, m)
    out.append("Checks passed: per-seed counts match Table I; sweep severity 0 equals seed-0 test result.")

    # Table I
    t1 = t1.copy()
    t1["test_hs_pct"] = (100 * t1.test_hs / (t1.test_hs + t1.test_nhs)).round(2)
    t1.to_csv(os.path.join(odir, "table1.csv"), index=False)

    # Table II: clean-data mean +/- std over seeds
    rows = []
    for b in B:
        for m in ("clean", "robust"):
            g = ps[(ps.benchmark == b) & (ps.model == m)]
            rows.append(dict(benchmark=b, model=m, bal_acc=ms(g.BalancedAccuracy), recall=ms(g.Recall),
                             specificity=ms(g.Specificity), precision=ms(g.Precision), f1=ms(g.F1),
                             false_alarms=f"{g.FP.mean():.0f} ± {g.FP.std(ddof=1):.0f}"))
    pd.DataFrame(rows).to_csv(os.path.join(odir, "table2.csv"), index=False)
    for m in ("clean", "robust"):
        avg = {k: np.mean([ps[(ps.benchmark == b) & (ps.model == m)][k].mean() for b in B])
               for k in ["BalancedAccuracy", "Recall", "Specificity", "Precision", "F1"]}
        out.append(f"Average over benchmarks ({m}): " + ", ".join(f"{k}={v:.3f}" for k, v in avg.items()))

    # Table III: robust-training ablation on clean data
    w = ps.pivot_table(index=["benchmark", "seed"], columns="model", values="BalancedAccuracy")
    d = (w.robust - w.clean).round(10)
    rows = []
    for b in B:
        db = d.loc[b]
        p = stats.wilcoxon(db).pvalue
        rows.append(dict(benchmark=b, mean_diff=f"{db.mean():+.4f}", robust_better=f"{int((db > 0).sum())}/3", wilcoxon_p=f"{p:.2f}"))
    pw = stats.wilcoxon(d).pvalue
    pt = stats.ttest_rel(w.robust, w.clean).pvalue
    rows.append(dict(benchmark="pooled (15 pairs)", mean_diff=f"{d.mean():+.4f}", robust_better=f"{int((d > 0).sum())}/15",
                     wilcoxon_p=f"{pw:.2f} (t-test {pt:.2f})"))
    pd.DataFrame(rows).to_csv(os.path.join(odir, "table3.csv"), index=False)
    out.append(f"Pooled clean-data comparison: mean diff {d.mean():+.4f}, robust better in {int((d > 0).sum())}/15, "
               f"Wilcoxon p = {pw:.2f}, paired t-test p = {pt:.2f}")

    # Table IV: balanced accuracy under perturbation
    rows = []
    for b in B:
        for m in ("clean", "robust"):
            g = sw[(sw.benchmark == b) & (sw.model == m)]
            cd = {s: g[(g.perturbation == "cd_bias") & (g.severity == s)].BalancedAccuracy.iloc[0] for s in range(5)}
            ep = {s: g[(g.perturbation == "epe") & (g.severity == s)].BalancedAccuracy.iloc[0] for s in range(5)}
            r = dict(benchmark=b, model=m, none=round(cd[0], 3))
            r.update({f"cd_k{s}": round(cd[s], 3) for s in (2, 3, 4)})
            r.update({f"epe_j{s}": round(ep[s], 3) for s in (1, 2, 3, 4)})
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(odir, "table4.csv"), index=False)

    # degradation relative to each model's own unperturbed accuracy
    for k, smin in (("cd_bias", 2), ("epe", 1)):
        t = sw[sw.perturbation == k].pivot_table(index=["benchmark", "severity"], columns="model", values="BalancedAccuracy")
        base = t.xs(0, level="severity")
        drop = t.sub(base, level="benchmark")
        per_sev = drop.groupby(level="severity").mean()
        out.append(f"{k}: mean change by severity (clean) {per_sev.clean.round(3).to_dict()}")
        out.append(f"{k}: mean change by severity (robust) {per_sev.robust.round(3).to_dict()}")
        drop = drop[drop.index.get_level_values("severity") >= smin]
        dd = drop.robust - drop.clean
        p = stats.wilcoxon(drop.robust, drop.clean).pvalue
        out.append(f"{k}: mean drop clean {drop.clean.mean():.3f}, robust {drop.robust.mean():.3f}; robust drops less in "
                   f"{int((dd > 0).sum())}/{len(dd)} cells; Wilcoxon p = {p:.2g} (descriptive, cells share models)")

    # non-monotonic steps
    ups = []
    for (b, m, k), g in sw.groupby(["benchmark", "model", "perturbation"]):
        ba = dict(zip(g.severity, g.BalancedAccuracy))
        fn = dict(zip(g.severity, g.FN))
        lv = [0, 2, 3, 4] if k == "cd_bias" else [0, 1, 2, 3, 4]
        for a, c in zip(lv, lv[1:]):
            if ba[c] - ba[a] > 1e-12:
                ups.append((round(ba[c] - ba[a], 4), b, m, k, a, c, fn[a], fn[c]))
    ups.sort()
    out.append(f"Increases with severity: {len(ups)} (<= 0.002: {sum(u[0] <= 0.002 for u in ups)})")
    out += [f"  +{u[0]:.4f} {u[1]} {u[2]} {u[3]} {u[4]}->{u[5]} missed hotspots {u[6]}->{u[7]}" for u in ups]

    # Table V: comparison with published results
    rows = []
    for b in B:
        r = dict(benchmark=b, published_recall=PUBLISHED[b][0], published_fa=PUBLISHED[b][1])
        for m in ("clean", "robust"):
            g = ps[(ps.benchmark == b) & (ps.model == m)]
            r[f"{m}_recall"] = round(100 * g.Recall.mean(), 1)
            r[f"{m}_fa"] = round(g.FP.mean())
        r["our_test_nhs"] = int(t1[t1.benchmark == b].test_nhs.iloc[0])
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(odir, "table5.csv"), index=False)

    # figures
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    cc, cr = "#6b6b6b", "#b03a2e"
    xl = {"cd_bias": "CD bias: kernel size k", "epe": "EPE: max patch shift j (px)"}
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    for i, k in enumerate(["cd_bias", "epe"]):
        for m, col, ls, lab in [("clean", cc, "--", "clean-trained"), ("robust", cr, "-", "robust-trained")]:
            t = sw[(sw.perturbation == k) & (sw.model == m)].pivot(index="benchmark", columns="severity", values="BalancedAccuracy")
            rel = t.sub(t[0], axis=0)
            for b in B:
                ax[i].plot(range(5), rel.loc[b], ls, color=col, lw=0.5, alpha=0.35)
            ax[i].plot(range(5), rel.mean(), ls, color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        ax[i].axhline(0, color="k", lw=0.4)
        ax[i].set_xticks(range(5))
        ax[i].set_xlabel(xl[k], fontsize=6.5)
        ax[i].grid(alpha=.25, lw=.4)
    ax[0].set_ylabel("Change in bal. acc.", fontsize=6.5)
    ax[0].legend(fontsize=5.5, frameon=False, loc="lower left")
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(odir, "fig1_mean_drop.png"), dpi=400)
    fig, ax = plt.subplots(2, 5, figsize=(7.1, 3.3))
    for r, k in enumerate(["cd_bias", "epe"]):
        for c, b in enumerate(B):
            for m, col, ls, lab in [("clean", cc, "--", "clean-trained"), ("robust", cr, "-", "robust-trained")]:
                g = sw[(sw.benchmark == b) & (sw.model == m) & (sw.perturbation == k)].sort_values("severity")
                ax[r, c].plot(g.severity, g.BalancedAccuracy, ls, color=col, marker="o", ms=2.2, lw=1.1, label=lab)
            ax[r, c].set_ylim(0.5, 1.0)
            ax[r, c].set_xticks(range(5))
            ax[r, c].grid(alpha=.25, lw=.4)
            if r == 0:
                ax[r, c].set_title(b, fontsize=7)
            if c == 0:
                ax[r, c].set_ylabel(("CD bias" if k == "cd_bias" else "EPE") + "\nbalanced accuracy", fontsize=6.5)
            else:
                ax[r, c].set_yticklabels([])
            ax[r, c].set_xlabel("k" if k == "cd_bias" else "j (px)", fontsize=6)
    ax[0, 0].legend(fontsize=5.2, loc="lower left", frameon=False)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(odir, "fig2_per_benchmark.png"), dpi=400)

    with open(os.path.join(odir, "stats.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "analysis_out")
