

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from math import radians, sin, cos, sqrt, atan2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family']        = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

DATASETS = [
    {
        "name":       "Beijing",
        "pkl":        Path("beijing_preprocessed_data_revise.pkl"),
        "data_dir":   Path(""),
        "geo_bounds": (39.8, 40.1, 116.3, 116.7),

        "grid_steps": (0.015, 0.015),
        "is_beijing": True,
    },
    {
        "name":       "Chengdu",
        "pkl":        Path("full_preprocessed_data_revise.pkl"),
        "data_dir":   Path("SiChuan"),
        "geo_bounds": (30.55, 30.75, 103.9, 104.2),
        "grid_steps": (0.015, 0.015),
        "is_beijing": False,
    },
]

SLOT_MINUTES = 15
OUTPUT_DIR   = Path("worker_quality_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_RECORDS_PER_SLOT = 3
MIN_DAYS_PER_CELL    = 3

ALPHA = 0.05

def load_data(cfg):
    path = cfg["data_dir"] / cfg["pkl"]
    with open(path, "rb") as f:
        cache = pickle.load(f)
    gps = cache["gps_data1"].copy()
    rdf = cache["result_df"].copy()
    gps["time"] = pd.to_datetime(gps["time"])
    rdf["date"] = pd.to_datetime(rdf["date"]).dt.date
    return gps, rdf

def prep_gps(gps, geo_bounds, grid_steps, sample_days,
             is_beijing=False):
    """
    ★ 与主实验对齐：
    - Beijing 没有 status 列，is_beijing=True 时不做 status 过滤
    - 网格步长已改为 0.015
    """
    lat_min, lat_max = geo_bounds[0], geo_bounds[1]
    lon_min, lon_max = geo_bounds[2], geo_bounds[3]
    lat_step, lon_step = grid_steps

    g = gps.dropna(subset=["time"]).copy()
    g["_date"] = g["time"].dt.date
    all_dates  = sorted(g["_date"].dropna().unique())
    if not all_dates:
        return pd.DataFrame(), []

    sample = all_dates[:sample_days]
    g      = g[g["_date"].isin(sample)].copy()
    g      = g[
        g["latitude"].between(lat_min, lat_max) &
        g["longitude"].between(lon_min, lon_max)
    ].copy()
    if g.empty:
        return pd.DataFrame(), sample

    g["grid_i"]    = ((g["latitude"]  - lat_min)
                      / lat_step).astype(int)
    g["grid_j"]    = ((g["longitude"] - lon_min)
                      / lon_step).astype(int)
    g["time_slot"] = (
        (g["time"].dt.hour * 60 + g["time"].dt.minute)
        // SLOT_MINUTES
    ).astype(int)
    return g, sample

def compute_sampling_sufficiency(gps, geo_bounds, grid_steps,
                                  sample_days=10,
                                  is_beijing=False):
    g, sample = prep_gps(gps, geo_bounds, grid_steps,
                         sample_days, is_beijing=is_beijing)
    if g.empty:
        return pd.DataFrame(
            columns=["taxi_id", "sampling_sufficiency"])

    print()

    counts = (g.groupby(
        ["taxi_id", "grid_i", "grid_j", "time_slot"])
        .size().reset_index(name="cnt"))

    med = (counts.groupby(["grid_i", "grid_j", "time_slot"])
           .agg(cell_med=("cnt", "median"),
                n_workers=("cnt", "count"))
           .reset_index())
    med = med[med["n_workers"] >= 2]

    merged = counts.merge(
        med, on=["grid_i", "grid_j", "time_slot"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["taxi_id", "sampling_sufficiency"])

    merged["above"] = (
        merged["cnt"] >= merged["cell_med"]).astype(float)
    score = (merged.groupby("taxi_id")["above"]
             .mean().reset_index(name="sampling_sufficiency"))

    print()
    return score

def compute_task_completion_consistency(gps, geo_bounds,
                                         grid_steps,
                                         sample_days=10,
                                         is_beijing=False):
    g, sample = prep_gps(gps, geo_bounds, grid_steps,
                         sample_days, is_beijing=is_beijing)
    if g.empty:
        return pd.DataFrame(
            columns=["taxi_id", "task_completion_consistency"])

    print()
    print()

    slot_counts = (
        g.groupby(["taxi_id", "_date", "time_slot"])
        .size().reset_index(name="slot_cnt"))

    slot_counts["sufficient"] = (
        slot_counts["slot_cnt"] >= MIN_RECORDS_PER_SLOT
    ).astype(float)

    daily = (slot_counts
             .groupby(["taxi_id", "_date"])
             .agg(n_slots     =("sufficient", "count"),
                  n_sufficient=("sufficient", "sum"))
             .reset_index())
    daily["daily_completion"] = (
        daily["n_sufficient"]
        / daily["n_slots"].clip(lower=1))

    score = (daily.groupby("taxi_id")["daily_completion"]
             .mean().reset_index(
                 name="task_completion_consistency"))

    print()
    return score

def compute_intra_cell_stability(gps, geo_bounds, grid_steps,
                                  sample_days=14,
                                  is_beijing=False):
    g, sample = prep_gps(gps, geo_bounds, grid_steps,
                         sample_days, is_beijing=is_beijing)
    if g.empty:
        return pd.DataFrame(
            columns=["taxi_id", "intra_cell_stability",
                     "n_cells_covered"])

    print()
    print()

    counts = (
        g.groupby(["taxi_id", "_date",
                   "grid_i", "grid_j", "time_slot"])
        .size().reset_index(name="cnt"))

    n_cells = (
        counts.groupby("taxi_id")
        [["grid_i", "grid_j", "time_slot"]]
        .apply(lambda df: len(df.drop_duplicates()))
        .reset_index(name="n_cells_covered"))

    cell_stats = (
        counts.groupby(
            ["taxi_id", "grid_i", "grid_j", "time_slot"])
        .agg(n_days  =("cnt", "count"),
             mean_cnt=("cnt", "mean"),
             std_cnt =("cnt", "std"))
        .reset_index())

    cell_stats = cell_stats[
        cell_stats["n_days"] >= MIN_DAYS_PER_CELL].copy()

    if cell_stats.empty:
        print()
        return pd.DataFrame(
            columns=["taxi_id", "intra_cell_stability",
                     "n_cells_covered"])

    cell_stats["cv"] = (
        cell_stats["std_cnt"].fillna(0)
        / (cell_stats["mean_cnt"] + 1e-8))

    worker_cv = (cell_stats
                 .groupby("taxi_id")
                 .agg(mean_cv=("cv", "mean"),
                      n_valid_cells=("cv", "count"))
                 .reset_index())
    worker_cv = worker_cv[
        worker_cv["n_valid_cells"] >= 2].copy()

    cv_min = worker_cv["mean_cv"].min()
    cv_max = worker_cv["mean_cv"].max()
    if cv_max - cv_min > 1e-6:
        worker_cv["intra_cell_stability"] = (
            1.0 - (worker_cv["mean_cv"] - cv_min)
            / (cv_max - cv_min))
    else:
        worker_cv["intra_cell_stability"] = 0.5

    worker_cv = worker_cv.merge(n_cells, on="taxi_id", how="left")

    print()
    return worker_cv[["taxi_id", "intra_cell_stability",
                       "n_cells_covered", "n_valid_cells"]]

def compute_behavioral_scores(rdf):
    cols = ["taxi_id"]
    if "wtcs" in rdf.columns:
        cols.append("wtcs")
    if "daily_reputation" in rdf.columns:
        cols.append("daily_reputation")
    agg = rdf[cols].groupby("taxi_id").mean().reset_index()

    if ("wtcs" in agg.columns
            and "daily_reputation" in agg.columns):
        agg["composite_score"] = (
            0.5 * agg["daily_reputation"]
            + 0.5 * agg["wtcs"])

    for col in ["wtcs", "daily_reputation", "composite_score"]:
        if col in agg.columns:
            lo = agg[col].min(); hi = agg[col].max()
            agg[f"{col}_norm"] = (
                (agg[col] - lo) / (hi - lo + 1e-8))
    return agg

def analyze_pair(x, y, x_label, y_label, dataset_name):
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n    = int(len(x))
    if n < 10:
        return {}

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    sig_p    = p_p < ALPHA
    sig_s    = p_s < ALPHA
    sig_both = sig_p and sig_s

    if sig_both:
        evidence = "strong"
        marker   = "*sig (both)*"
    elif sig_p:
        evidence = "partial-Pearson"
        marker   = "! Pearson-only (Spearman n.s.)"
    elif sig_s:
        evidence = "partial-Spearman"
        marker   = "! Spearman-only (Pearson n.s.)"
    else:
        evidence = "none"
        marker   = "n.s."

    print()
    print()
    print()
    print()

    return {
        "dataset":               dataset_name,
        "x_label":               x_label,
        "y_label":               y_label,
        "n":                     n,
        "pearson_r":             round(float(r_p), 4),
        "pearson_p":             round(float(p_p), 4),
        "spearman_r":            round(float(r_s), 4),
        "spearman_p":            round(float(p_s), 4),
        "direction":             "positive" if r_p > 0 else "negative",
        "significant_both":      bool(sig_both),
        "significant_pearson":   bool(sig_p),
        "significant_spearman":  bool(sig_s),
        "evidence_strength":     evidence,
        "x_mean":                round(float(x.mean()), 4),
        "y_mean":                round(float(y.mean()), 4),
    }

def verify_causal_hypothesis(merged, dataset_name):
    if ("wtcs_norm" not in merged.columns
            or "n_cells_covered" not in merged.columns):
        print()
        return None

    sub1 = merged[["wtcs_norm", "n_cells_covered"]].dropna()
    if len(sub1) < 10:
        return None

    r1_p, p1_p = pearsonr(sub1["wtcs_norm"].values,
                           sub1["n_cells_covered"].values)
    r1_s, p1_s = spearmanr(sub1["wtcs_norm"].values,
                            sub1["n_cells_covered"].values)
    sig1 = (p1_p < ALPHA) and (p1_s < ALPHA) and (r1_p > 0)

    print()
    print()
    print()
    print()

    link2_result = None
    if "intra_cell_stability" in merged.columns:
        sub2 = merged[
            ["n_cells_covered", "intra_cell_stability"]
        ].dropna()

        if len(sub2) >= 10:
            r2_p, p2_p = pearsonr(
                sub2["n_cells_covered"].values,
                sub2["intra_cell_stability"].values)
            r2_s, p2_s = spearmanr(
                sub2["n_cells_covered"].values,
                sub2["intra_cell_stability"].values)

            sig2 = (p2_p < ALPHA) and (p2_s < ALPHA) and (r2_p < 0)

            print()
            print()
            print()

            link2_result = {
                "r_p": round(float(r2_p), 4),
                "p_p": round(float(p2_p), 4),
                "r_s": round(float(r2_s), 4),
                "p_s": round(float(p2_s), 4),
                "sig_negative": bool(sig2),
                "n": len(sub2),
            }

            if sig2:
                print()
            else:
                print()
        else:
            print()
    else:
        print()

    chain_complete = bool(
        sig1
        and link2_result is not None
        and link2_result["sig_negative"]
    )

    if chain_complete:
        print()
    else:
        print()

    return {
        "dataset":        dataset_name,
        "link1_r_p":      round(float(r1_p), 4),
        "link1_p_p":      round(float(p1_p), 4),
        "link1_r_s":      round(float(r1_s), 4),
        "link1_p_s":      round(float(p1_s), 4),
        "link1_sig":      bool(sig1),
        "link1_n":        len(sub1),
        "link2":          link2_result,
        "chain_complete": chain_complete,
        "pearson_r":      round(float(r1_p), 4),
        "pearson_p":      round(float(p1_p), 4),
        "causal_supported": chain_complete,
        "n":              len(sub1),
    }

def run_correlations(behavioral, suf_df, comp_df,
                     stab_df, dataset_name):
    merged = behavioral.copy()
    for df in [suf_df, comp_df, stab_df]:
        if df is not None and not df.empty:
            keep = [c for c in df.columns
                    if c not in merged.columns or c == "taxi_id"]
            merged = merged.merge(df[keep],
                                  on="taxi_id", how="left")

    print()

    n_beh  = behavioral["taxi_id"].nunique()
    n_suf  = (suf_df["taxi_id"].nunique()
              if suf_df is not None and not suf_df.empty else 0)
    n_comp = (comp_df["taxi_id"].nunique()
              if comp_df is not None and not comp_df.empty else 0)
    n_stab = (stab_df["taxi_id"].nunique()
              if stab_df is not None and not stab_df.empty else 0)
    print()
    print()
    print()
    print()

    beh_cols = [
        ("wtcs_norm",             "WTCS"),
        ("daily_reputation_norm", "Reputation"),
    ]
    data_cols = [
        ("sampling_sufficiency",        "Sampling Sufficiency"),
        ("task_completion_consistency",  "Task Completion"),
        ("intra_cell_stability",        "Intra-cell Stability"),
    ]

    results = []
    for bc, bl in beh_cols:
        if bc not in merged.columns:
            continue
        for dc, dl in data_cols:
            if dc not in merged.columns:
                continue
            sub = merged[[bc, dc]].dropna()
            if len(sub) < 10:
                continue
            res = analyze_pair(
                sub[bc].values, sub[dc].values,
                bl, dl, dataset_name)
            if res:
                results.append(res)

    causal = verify_causal_hypothesis(merged, dataset_name)

    out = OUTPUT_DIR / f"{dataset_name.lower()}_quality_merged.csv"
    merged.to_csv(out, index=False)
    print()

    return results, merged, causal

def print_interpretation(all_corr, causal_results):
    print()
    print()
    print()

    strong  = [r for r in all_corr
               if r.get("evidence_strength") == "strong"]
    partial = [r for r in all_corr
               if r.get("evidence_strength", "").startswith("partial")]
    non_sig = [r for r in all_corr
               if r.get("evidence_strength") == "none"]

    print()
    for r in strong:
        dirn = "正相关" if r["direction"] == "positive" else "负相关"
        print()

    print()
    for r in partial:
        evd   = r.get("evidence_strength", "")
        which = ("Pearson 显著，Spearman n.s."
                 if "Pearson" in evd
                 else "Spearman 显著，Pearson n.s.")
        dirn = "正相关" if r["direction"] == "positive" else "负相关"
        print()

    print()
    for r in non_sig:
        print()

    print()
    for c in (causal_results or []):
        if c is None:
            continue
        chain = c.get("chain_complete", False)
        l2    = c.get("link2")
        print()
        print()
        if l2:
            print()
        print()

def print_latex_table(all_corr, causal_results):
    print()
    print()
    print()

    n_by_indicator = {}
    for res in all_corr:
        key = (res["dataset"], res["y_label"])
        n_by_indicator.setdefault(key, []).append(res["n"])

    intra_notes_parts = []
    other_notes_seen  = set()
    for (ds, yl), ns in n_by_indicator.items():
        n_val = ns[0]
        if "Intra" in yl:
            intra_notes_parts.append(f"{ds}: $n={n_val}$")
        else:
            other_notes_seen.add(f"{ds}: $n={n_val}$")
    intra_n_str = "; ".join(intra_notes_parts)
    other_n_str = "; ".join(sorted(other_notes_seen))

    causal_note_parts = []
    for c in (causal_results or []):
        if c is None:
            continue
        ds    = c.get("dataset", "")
        l1r   = c.get("link1_r_p", 0.0)
        l2    = c.get("link2")
        chain = c.get("chain_complete", False)
        if chain and l2:
            l2r = l2.get("r_p", 0.0)
            causal_note_parts.append(
                f"{ds}: confirmed "
                f"($r_{{\\text{{WTCS}}\\to\\text{{cells}}}}={l1r:+.3f}$; "
                f"$r_{{\\text{{cells}}\\to\\text{{IDS}}}}={l2r:.3f}$)")
        elif c.get("link1_sig"):
            causal_note_parts.append(
                f"{ds}: first link confirmed "
                f"($r={l1r:+.3f}$); second link inconclusive")
        else:
            causal_note_parts.append(
                f"{ds}: not confirmed")
    causal_note_str = " ".join(causal_note_parts)

    latex = r"""% \usepackage{threeparttable}
\begin{table}[htbp]
\caption{Correlation between behavioral compliance metrics 
(WTCS and Reputation) and three data quality indicators. 
Significance: $^{*}$ both Pearson and Spearman $p < 0.05$; 
$^{\dagger}$ only one test $p < 0.05$.}
\label{tab:quality_correlation}
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{threeparttable}
\begin{tabular}{lllrcccc}
\hline
\textbf{Dataset} 
  & \textbf{Behavioral} 
  & \textbf{Quality Indicator}
  & \textbf{$n$}
  & \textbf{Pearson $r$} & \textbf{$p$} 
  & \textbf{Spearman $\rho$} & \textbf{$p$} \\
\hline
"""

    prev_ds = None
    for res in all_corr:
        ds  = res.get("dataset",    "")
        xl  = res.get("x_label",   "")
        yl  = res.get("y_label",   "")
        n   = res.get("n",          "")
        pr  = res.get("pearson_r",  0.0)
        pp  = res.get("pearson_p",  1.0)
        sr  = res.get("spearman_r", 0.0)
        sp  = res.get("spearman_p", 1.0)
        evd = res.get("evidence_strength", "none")

        if evd == "strong":
            mark_p = r"$^{*}$"
            mark_s = r"$^{*}$"
        elif evd == "partial-Pearson":
            mark_p = r"$^{*}$"
            mark_s = r"$^{\dagger}$"
        elif evd == "partial-Spearman":
            mark_p = r"$^{\dagger}$"
            mark_s = r"$^{*}$"
        else:
            mark_p = ""
            mark_s = ""

        ds_str  = ds if ds != prev_ds else ""
        prev_ds = ds

        latex += (
            f"{ds_str} & {xl} & {yl} & {n}"
            f" & {pr:+.3f}{mark_p} & {pp:.3f}"
            f" & {sr:+.3f}{mark_s} & {sp:.3f} \\\\\n"
        )

    latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\footnotesize
\item[$*$] Both tests $p < 0.05$ (strong evidence).
\item[$\dagger$] One test $p < 0.05$ (partial evidence).
\item[IDS] Intra-cell Stability. """ + f"Sample sizes: {intra_n_str}" + r""" 
  (other indicators: """ + other_n_str + r""").
"""

    if causal_note_str:
        latex += (
            r"\item[C] Causal chain (WTCS$\to$coverage$\to$IDS): "
            f"{causal_note_str}\n"
        )

    latex += r"""\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    print()

def plot_scatter_grid(merged, dataset_name, corr_results):
    beh_cols  = [("wtcs_norm",             "WTCS"),
                 ("daily_reputation_norm", "Reputation")]
    data_cols = [("sampling_sufficiency",        "Sampling Sufficiency"),
                 ("task_completion_consistency",  "Task Completion"),
                 ("intra_cell_stability",         "Intra-cell Stability")]

    avail_beh  = [(c, l) for c, l in beh_cols
                  if c in merged.columns]
    avail_data = [(c, l) for c, l in data_cols
                  if c in merged.columns]
    if not avail_beh or not avail_data:
        return

    nrow = len(avail_beh); ncol = len(avail_data)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow))
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    def get_res(xl, yl):
        for r in corr_results:
            if (r.get("x_label") == xl
                    and r.get("y_label") == yl
                    and r.get("dataset") == dataset_name):
                return r
        return None

    colors_by_evd = {
        "strong":           "#0072BD",
        "partial-Pearson":  "#D95319",
        "partial-Spearman": "#D95319",
        "none":             "#AAAAAA",
    }

    for i, (bc, bl) in enumerate(avail_beh):
        for j, (dc, dl) in enumerate(avail_data):
            ax  = axes[i, j]
            sub = merged[[bc, dc]].dropna()
            if sub.empty:
                ax.set_visible(False); continue

            res    = get_res(bl, dl)
            evd    = res.get("evidence_strength",
                             "none") if res else "none"
            pt_col = colors_by_evd.get(evd, "#AAAAAA")

            x, y = sub[bc].values, sub[dc].values
            ax.scatter(x, y, alpha=0.35, s=18,
                       color=pt_col, edgecolors="none")

            if len(x) > 2:
                z  = np.polyfit(x, y, 1)
                xr = np.linspace(x.min(), x.max(), 100)
                ax.plot(xr, np.poly1d(z)(xr),
                        color="black", linewidth=1.6,
                        linestyle="--")

            if res:
                r_p = res["pearson_r"]
                p_p = res["pearson_p"]
                r_s = res["spearman_r"]
                p_s = res["spearman_p"]
                n   = res["n"]
                if evd == "strong":
                    sig_note = f"r={r_p:+.3f}*, ρ={r_s:+.3f}*"
                elif evd.startswith("partial"):
                    sig_note = (
                        f"r={r_p:+.3f}{'*' if p_p<ALPHA else ''}, "
                        f"ρ={r_s:+.3f}{'*' if p_s<ALPHA else ''}†")
                else:
                    sig_note = f"r={r_p:+.3f} n.s."
                ax.set_title(
                    f"{bl} vs. {dl}  [n={n}]\n{sig_note}",
                    fontsize=9.5)

            ax.set_xlabel(bl, fontsize=10)
            ax.set_ylabel(dl, fontsize=10)
            ax.tick_params(direction="in", top=True, right=True)
            ax.grid(True, linestyle="--", alpha=0.2)
            for sp in ax.spines.values():
                sp.set_edgecolor("black"); sp.set_linewidth(1.0)

    fig.suptitle(
        f"{dataset_name}: Behavioral Compliance vs. Data Quality",
        fontsize=11, y=1.01)
    plt.tight_layout()
    fn = OUTPUT_DIR / f"scatter_grid_{dataset_name.lower()}.pdf"
    plt.savefig(fn, bbox_inches="tight")
    plt.savefig(str(fn).replace(".pdf", ".png"),
                dpi=300, bbox_inches="tight")
    print()
    plt.close()

def plot_boxplot(merged, dataset_name):
    data_cols = [
        ("sampling_sufficiency",        "Sampling Sufficiency"),
        ("task_completion_consistency",  "Task Completion"),
        ("intra_cell_stability",         "Intra-cell Stability"),
    ]
    beh_cols = [
        ("wtcs_norm",             "WTCS"),
        ("daily_reputation_norm", "Reputation"),
    ]
    avail_data = [(c, l) for c, l in data_cols
                  if c in merged.columns]
    avail_beh  = [(c, l) for c, l in beh_cols
                  if c in merged.columns]
    if not avail_data or not avail_beh:
        return

    nrow = len(avail_beh); ncol = len(avail_data)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.8 * ncol, 4.0 * nrow))
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    mpl_ver = tuple(
        int(x) for x in
        matplotlib.__version__.split(".")[:2])
    colors = ["#EDB120", "#0072BD", "#7E2F8E"]

    for i, (bc, bl) in enumerate(avail_beh):
        for j, (dc, dl) in enumerate(avail_data):
            ax  = axes[i, j]
            sub = merged[[bc, dc]].dropna().copy()
            if len(sub) < 6:
                ax.set_visible(False); continue

            sub["grp"] = pd.qcut(
                sub[bc], q=3,
                labels=["Low", "Medium", "High"],
                duplicates="drop")
            groups = [
                sub[sub["grp"] == g][dc].values
                for g in ["Low", "Medium", "High"]
            ]
            bp_kw = dict(
                patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
            if mpl_ver >= (3, 9):
                bp_kw["tick_labels"] = ["Low", "Medium", "High"]
            else:
                bp_kw["labels"]      = ["Low", "Medium", "High"]

            bp = ax.boxplot(groups, **bp_kw)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.75)

            ax.set_xlabel(f"{bl} Group", fontsize=10)
            ax.set_ylabel(dl, fontsize=10)
            ax.set_title(
                f"{dataset_name}: {bl} → {dl}  [n={len(sub)}]",
                fontsize=9.5)
            ax.tick_params(direction="in", top=True, right=True)
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            for sp in ax.spines.values():
                sp.set_edgecolor("black")

    plt.tight_layout()
    fn = (OUTPUT_DIR
          / f"boxplot_{dataset_name.lower()}_quality.pdf")
    plt.savefig(fn, bbox_inches="tight")
    plt.savefig(str(fn).replace(".pdf", ".png"),
                dpi=300, bbox_inches="tight")
    print()
    plt.close()

def main():
    print()
    print()
    print()
    print()
    print()

    all_corr   = []
    all_causal = []

    for cfg in DATASETS:
        name       = cfg["name"]
        is_beijing = cfg.get("is_beijing", False)

        print()
        print()
        print()

        try:
            gps, rdf = load_data(cfg)
        except FileNotFoundError as e:
            print()
            continue

        print()
        behavioral = compute_behavioral_scores(rdf)
        print()
        for col in ["wtcs", "daily_reputation"]:
            if col in behavioral.columns:
                s = behavioral[col]
                print()

        print()
        suf_df  = compute_sampling_sufficiency(
            gps, cfg["geo_bounds"], cfg["grid_steps"],
            sample_days=10, is_beijing=is_beijing)

        print()
        comp_df = compute_task_completion_consistency(
            gps, cfg["geo_bounds"], cfg["grid_steps"],
            sample_days=10, is_beijing=is_beijing)

        print()
        stab_df = compute_intra_cell_stability(
            gps, cfg["geo_bounds"], cfg["grid_steps"],
            sample_days=14, is_beijing=is_beijing)

        print()
        corr_results, merged, causal = run_correlations(
            behavioral, suf_df, comp_df, stab_df, name)
        all_corr.extend(corr_results)
        all_causal.append(causal)

        print()
        plot_scatter_grid(merged, name, all_corr)
        plot_boxplot(merged, name)

    if all_corr:
        df_sum = pd.DataFrame(all_corr)
        out    = OUTPUT_DIR / "quality_correlation_summary.csv"
        df_sum.to_csv(out, index=False)
        print()

        key_cols = [
            "dataset", "x_label", "y_label", "n",
            "pearson_r", "pearson_p",
            "spearman_r", "spearman_p",
            "evidence_strength",
        ]
        print()

        print_latex_table(all_corr, all_causal)
        print_interpretation(all_corr, all_causal)

    print()

if __name__ == "__main__":
    main()