# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loading, scoring and plotting of classification results for the results notebooks.

Every function works on whatever results exist: missing grids, thresholds,
feature sets or schemes are skipped (with a note) rather than raising.

Score: chance-normalized balanced accuracy, (bacc - 1/K) / (1 - 1/K), with K the
number of classes of the problem: 0 = chance, 1 = perfect.  One value per
classification problem (mouse x label within mouse; held-out mouse or mouse pair
across mice); 3D-CNN repeats are averaged per problem first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from results_index import load_metrics  # noqa: E402

FIG_DIR = PROJECT_DIR / "notebooks" / "figures" / "results"

# ---------------------------------------------------------------------------
# Style (reference palette: first three categorical slots, validated all-pairs)
# ---------------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_LINE = "#e3e2de"
FAMILY_COLORS = {
    "3D-CNN": "#1baf7a",  # aqua
    "topology": "#2a78d6",  # blue
    "level": "#eb6834",  # orange
    "level (trial avg)": "#eb6834",  # orange, hollow markers
}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID_LINE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)

SCHEMES = [
    ("video_label", "within_mouse"),
    ("video_label", "cross_mouse"),
    ("video_id", "within_mouse"),
    ("video_id", "cross_mouse"),
    ("segment_id", "within_mouse"),
    ("segment_id", "cross_mouse"),
]
SCHEME_TITLES = {
    ("video_label", "within_mouse"): "Video label, within mouse",
    ("video_label", "cross_mouse"): "Video label, cross mouse",
    ("video_id", "within_mouse"): "Video ID, within mouse",
    ("video_id", "cross_mouse"): "Video ID, cross mouse",
    ("segment_id", "within_mouse"): "Segment ID, within mouse",
    ("segment_id", "cross_mouse"): "Segment ID, cross mouse",
}
GRID_TITLES = {
    "grid-15x15x10_norm-by_minmax": "min-max grid",
    "grid-15x15x10_norm-by_trialmean": "trial-mean grid",
}
SCORE_LABEL = "chance-normalized balanced accuracy"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _thr_label(row: pd.Series) -> str:
    if row.get("thresh_mode") == "abs":
        t = row.get("abs_threshold")
        return f"abs{0 if pd.isna(t) else float(t):g}"
    p = row.get("p_active")
    return "" if pd.isna(p) else f"p{int(p)}"


def _thr_sort_key(thr: str) -> tuple:
    if thr.startswith("abs"):
        return (0, float(thr[3:]))
    if thr.startswith("p"):
        return (1, int(thr[1:]))
    return (2, 0)


def _feature_info(row: pd.Series) -> tuple:
    """(feature name without threshold, family) of a result row."""
    if row["input"] == "grid":
        return "3D-CNN", "3D-CNN"
    vec = str(row["vectorization"])
    if row["input"] == "zigzag":
        name = vec.replace("+active-fraction", " + AF")
        return name, "topology"
    # grid statistics
    trialavg = vec.endswith("-trialavg")
    base = vec[: -len("-trialavg")] if trialavg else vec
    names = {
        "active-fraction": "AF",
        "moments-raw": "moments",
        "moments-std": "moments",
        "moments-density": "moments (dens.)",
        "moments-std-density": "moments (dens.)",
    }
    name = names.get(base, base)
    if trialavg:
        return f"{name}, avg", "level (trial avg)"
    return name, "level"


def load_all(include_partial: bool = False, results_root: Optional[Path] = None) -> pd.DataFrame:
    """All results (latest run per configuration) with score and labels added.

    Columns added: grid, thr, feature, family, label (feature + threshold),
    scheme (target, scope), score; 3D-CNN repeats are kept as rows (see
    ``per_problem`` to average them).
    """
    kwargs = {"latest_run_only": True, "include_partial": include_partial}
    if results_root is not None:
        kwargs["results_root"] = results_root
    df = load_metrics(**kwargs)
    if df.empty:
        return df
    df = df.copy()
    df["grid"] = df["grid_type"].map(GRID_TITLES).fillna(df["grid_type"])
    df["thr"] = df.apply(_thr_label, axis=1)
    info = df.apply(_feature_info, axis=1, result_type="expand")
    df["feature"], df["family"] = info[0], info[1]
    df["label"] = np.where(df["thr"] != "", df["feature"] + " " + df["thr"], df["feature"])
    df["scheme"] = list(zip(df["target"], df["scope"]))
    k = df["n_classes"].astype(float)
    df["score"] = (df["balanced_accuracy"] - 1.0 / k) / (1.0 - 1.0 / k)
    return df


def per_problem(df: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    """One row per (grid, feature, threshold, scheme, problem); repeats averaged."""
    keys = ["grid", "grid_type", "feature", "family", "thr", "label", "target", "scope", "unit_id", "mouse", "n_classes"]
    keys = [k for k in keys if k in df.columns]
    out = df.groupby(keys, dropna=False).agg(
        **{value: (value, "mean"), f"{value}_sd": (value, "std"), "n_repeats": (value, "size")}
    )
    return out.reset_index()


def thresholds(df: pd.DataFrame) -> List[str]:
    return sorted({t for t in df["thr"].unique() if t}, key=_thr_sort_key)


def feature_order(df: pd.DataFrame) -> List[str]:
    """Display order of labels: 3D-CNN, then Turnover / AF per threshold, then moments."""
    labels = set(df["label"])
    order: List[str] = []
    if "3D-CNN" in labels:
        order.append("3D-CNN")
    for thr in thresholds(df):
        for feat in ["Turnover", "AF"]:
            lab = f"{feat} {thr}"
            if lab in labels:
                order.append(lab)
    for lab in ["moments", "moments (dens.)", "moments, avg", "moments (dens.), avg"]:
        if lab in labels:
            order.append(lab)
    return order


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def coverage(df_all: pd.DataFrame) -> pd.DataFrame:
    """Problems per (grid, feature set) x scheme; '(partial)' marks unfinished runs, '—' missing."""
    if df_all.empty:
        return pd.DataFrame()
    d = df_all.copy()
    d["scheme_title"] = d["scheme"].map(SCHEME_TITLES)
    d["partial"] = (d["run_status"] != "complete") if "run_status" in d else False
    agg = d.groupby(["grid", "family", "label", "scheme_title"]).agg(
        problems=("unit_id", "nunique"), partial=("partial", "any")
    ).reset_index()
    agg["cell"] = agg["problems"].astype(str) + np.where(agg["partial"], " (partial)", "")
    table = agg.pivot_table(index=["grid", "family", "label"], columns="scheme_title", values="cell", aggfunc="first")
    cols = [SCHEME_TITLES[s] for s in SCHEMES if SCHEME_TITLES[s] in table.columns]
    return table.reindex(columns=cols).fillna("—")


def sample_count_check(df_all: pd.DataFrame) -> pd.DataFrame:
    """Problems whose number of samples differs between inputs (should be empty)."""
    keys = ["grid", "target", "scope", "unit_id"]
    n = df_all.groupby(keys)["n_samples"].nunique()
    bad = n[n > 1].index
    if len(bad) == 0:
        return pd.DataFrame(columns=keys + ["n_samples_values"])
    sub = df_all.set_index(keys).loc[bad].reset_index()
    return sub.groupby(keys)["n_samples"].apply(lambda s: sorted(set(s))).rename("n_samples_values").reset_index()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _empty_panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "no results yet", ha="center", va="center", color=INK_2, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _dots_and_mean(ax, x: float, values: np.ndarray, color: str, hollow: bool = False, rng=None) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    rng = rng or np.random.default_rng(0)
    jitter = rng.uniform(-0.18, 0.18, size=values.size)
    face = "white" if hollow else color
    ax.scatter(x + jitter, values, s=12, facecolor=face, edgecolor=color, linewidth=0.8, alpha=0.55, zorder=2)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
    ax.errorbar(x + 0.32, mean, yerr=sem, fmt="o", ms=5, color=color, mfc=face, mec=color, mew=1.2, elinewidth=1.4,
                capsize=0, zorder=3)


def _family_legend(fig: plt.Figure, families: Iterable[str]) -> None:
    handles = []
    for fam in families:
        color = FAMILY_COLORS[fam]
        hollow = "trial avg" in fam
        handles.append(Line2D([], [], marker="o", ls="", color=color, mfc="white" if hollow else color, mec=color, label=fam))
    handles.append(Line2D([], [], marker="o", ls="", color=INK_2, label="dots: problems; marker: mean ± s.e.m."))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))


# ---------------------------------------------------------------------------
# 01: performance overview
# ---------------------------------------------------------------------------

def plot_overview(pp: pd.DataFrame, grid: str, value: str = "score") -> Optional[plt.Figure]:
    """Score per feature set for one grid, one panel per scheme."""
    d = pp[pp["grid"] == grid]
    if d.empty:
        print(f"{grid}: no results yet")
        return None
    order = feature_order(d)
    # Each panel has its own y-range: scores differ a lot between schemes.
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    rng = np.random.default_rng(0)
    for ax, scheme in zip(axes.ravel(), SCHEMES):
        sd = d[(d["target"] == scheme[0]) & (d["scope"] == scheme[1])]
        if sd.empty:
            _empty_panel(ax, SCHEME_TITLES[scheme])
            continue
        labs = [l for l in order if l in set(sd["label"])]
        for i, lab in enumerate(labs):
            rows = sd[sd["label"] == lab]
            fam = rows["family"].iloc[0]
            _dots_and_mean(ax, i, rows[value].to_numpy(), FAMILY_COLORS[fam], hollow="trial avg" in fam, rng=rng)
        ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=50, ha="right")
        ax.set_xlim(-0.6, len(labs) - 0.2)
        ax.set_title(f"{SCHEME_TITLES[scheme]}  (n problems = {sd['unit_id'].nunique()})")
    for ax in axes[:, 0]:
        ax.set_ylabel(SCORE_LABEL if value == "score" else value)
    fams = [f for f in FAMILY_COLORS if f in set(d["family"])]
    _family_legend(fig, fams)
    fig.suptitle(f"Classification performance, {grid} (0 = chance)", y=1.03, color=INK)
    fig.tight_layout()
    return fig


def overview_table(pp: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    """Mean score per (grid, feature set) x scheme."""
    if pp.empty:
        return pd.DataFrame()
    t = pp.groupby(["grid", "label", "target", "scope"])[value].mean().reset_index()
    t["scheme"] = [SCHEME_TITLES[(a, b)] for a, b in zip(t["target"], t["scope"])]
    table = t.pivot_table(index=["grid", "label"], columns="scheme", values=value)
    cols = [SCHEME_TITLES[s] for s in SCHEMES if SCHEME_TITLES[s] in table.columns]
    return table[cols]


def plot_heatmap(pp: pd.DataFrame, grid: str, value: str = "score") -> Optional[plt.Figure]:
    """Mean score, feature set x scheme, for one grid."""
    d = pp[pp["grid"] == grid]
    if d.empty:
        print(f"{grid}: no results yet")
        return None
    table = overview_table(d, value).loc[grid]
    order = [l for l in feature_order(d) if l in table.index]
    table = table.loc[order]
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 3, 0.38 * table.shape[0] + 1.5))
    vals = table.to_numpy(dtype=float)
    im = ax.imshow(np.clip(vals, 0, 1), cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(vals):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"mean {SCORE_LABEL}", fraction=0.03)
    ax.set_title(f"Mean performance, {grid} (blank = no results)")
    fig.tight_layout()
    return fig


def cnn_repeat_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spread of the 3D-CNN score across repeats, per grid and scheme."""
    d = df[df["input"] == "grid"]
    if d.empty:
        return pd.DataFrame()
    per = d.groupby(["grid", "target", "scope", "unit_id"])["score"].agg(["std", "size"]).reset_index()
    per = per[per["size"] > 1]
    if per.empty:
        return pd.DataFrame()
    return per.groupby(["grid", "target", "scope"]).agg(
        problems=("unit_id", "nunique"), repeats=("size", "max"), mean_sd_across_repeats=("std", "mean"),
        max_sd_across_repeats=("std", "max"),
    )


# ---------------------------------------------------------------------------
# Paired comparisons and statistics
# ---------------------------------------------------------------------------

def paired(pp: pd.DataFrame, a: Dict[str, object], b: Dict[str, object], keys: Sequence[str]) -> pd.DataFrame:
    """Per-problem difference a - b, matched on ``keys`` (e.g. grid, scheme, unit_id).

    ``a`` and ``b`` select rows by column values (e.g. {"feature": "Turnover + AF"}).
    """
    def sel(spec):
        m = np.ones(len(pp), dtype=bool)
        for k, v in spec.items():
            m &= (pp[k] == v).to_numpy()
        return pp[m]

    da, db = sel(a), sel(b)
    if da.empty or db.empty:
        return pd.DataFrame()
    m = da.merge(db, on=list(keys), suffixes=("_a", "_b"))
    if m.empty:
        return m
    m["delta"] = m["score_a"] - m["score_b"]
    return m


def wilcoxon_table(rows: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Wilcoxon signed-rank test of delta != 0 per group, Holm-corrected over all groups.

    With fewer than 6 problems a two-sided test cannot reach p < 0.05; those
    groups are reported with their mean difference only.
    """
    from scipy.stats import wilcoxon

    out = []
    for key, g in rows.groupby(list(group_cols), dropna=False):
        dlt = g["delta"].dropna().to_numpy()
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update({"n": len(dlt), "mean_delta": dlt.mean() if len(dlt) else np.nan,
                    "median_delta": np.median(dlt) if len(dlt) else np.nan,
                    "frac_positive": (dlt > 0).mean() if len(dlt) else np.nan, "p": np.nan})
        if len(dlt) >= 6 and np.any(dlt != 0):
            rec["p"] = wilcoxon(dlt).pvalue
        out.append(rec)
    res = pd.DataFrame(out)
    if res.empty:
        return res
    ok = res["p"].notna()
    res["p_holm"] = np.nan
    if ok.any():
        p = res.loc[ok, "p"].to_numpy()
        order = np.argsort(p)
        adj = np.empty_like(p)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(p) - rank) * p[idx])
            adj[idx] = min(running, 1.0)
        res.loc[ok, "p_holm"] = adj
    return res


# ---------------------------------------------------------------------------
# 02: incremental information
# ---------------------------------------------------------------------------

SUBSETS = [("Turnover", "all dims"), ("Turnover-H1", "H1"), ("Turnover-H2", "H2")]


def incremental_deltas(pp: pd.DataFrame) -> pd.DataFrame:
    """Per problem: (topology subset + AF) - AF, and topology subset alone - AF."""
    rows = []
    keys = ["grid", "thr", "target", "scope", "unit_id"]
    for feat, subset in SUBSETS:
        for kind, a_feat in (("topology + AF  vs  AF", f"{feat} + AF"), ("topology alone  vs  AF", feat)):
            m = paired(pp, {"feature": a_feat, "family": "topology"}, {"feature": "AF", "family": "level"}, keys)
            if m.empty:
                continue
            m = m[keys + ["delta", "score_a", "score_b"]].copy()
            m["subset"], m["comparison"] = subset, kind
            rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_incremental(deltas: pd.DataFrame, grid: str, comparison: str = "topology + AF  vs  AF") -> Optional[plt.Figure]:
    """Per scheme (rows) and threshold (columns): delta per topology subset."""
    d = deltas[(deltas["grid"] == grid) & (deltas["comparison"] == comparison)] if not deltas.empty else deltas
    if d.empty:
        print(f"{grid}: no paired results for '{comparison}' yet")
        return None
    thrs = sorted(d["thr"].unique(), key=_thr_sort_key)
    fig, axes = plt.subplots(len(SCHEMES), len(thrs), figsize=(2.6 * len(thrs) + 1.5, 2.1 * len(SCHEMES)),
                             sharey="row", squeeze=False)
    rng = np.random.default_rng(0)
    subsets = [s for _f, s in SUBSETS]
    for r, scheme in enumerate(SCHEMES):
        for c, thr in enumerate(thrs):
            ax = axes[r, c]
            sd = d[(d["target"] == scheme[0]) & (d["scope"] == scheme[1]) & (d["thr"] == thr)]
            if sd.empty:
                _empty_panel(ax, thr if r == 0 else "")
                continue
            for i, sub in enumerate(subsets):
                _dots_and_mean(ax, i, sd.loc[sd["subset"] == sub, "delta"].to_numpy(), FAMILY_COLORS["topology"], rng=rng)
            ax.axhline(0, color=INK_2, lw=0.9, zorder=1)
            ax.set_xticks(range(len(subsets)))
            ax.set_xticklabels(subsets if r == len(SCHEMES) - 1 else [])
            ax.set_xlim(-0.6, len(subsets) - 0.2)
            if r == 0:
                ax.set_title(thr)
            if c == 0:
                ax.set_ylabel(SCHEME_TITLES[scheme].replace(", ", "\n"), fontsize=8)
    fig.suptitle(f"{comparison}: difference in {SCORE_LABEL}, {grid}\n(dots: problems; marker: mean ± s.e.m.; > 0: topology adds information)",
                 y=1.01, color=INK)
    fig.tight_layout()
    return fig


def plot_components(pp: pd.DataFrame, grid: str, thr: str) -> Optional[plt.Figure]:
    """AF alone, topology alone and topology + AF, per scheme, for one grid and threshold."""
    d = pp[(pp["grid"] == grid) & (pp["thr"] == thr)]
    labs = ["AF", "Turnover", "Turnover + AF", "Turnover-H1", "Turnover-H1 + AF", "Turnover-H2", "Turnover-H2 + AF"]
    d = d[d["feature"].isin(labs)]
    if d.empty:
        print(f"{grid} {thr}: no results yet")
        return None
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    rng = np.random.default_rng(0)
    for ax, scheme in zip(axes.ravel(), SCHEMES):
        sd = d[(d["target"] == scheme[0]) & (d["scope"] == scheme[1])]
        present = [l for l in labs if l in set(sd["feature"])]
        if not present:
            _empty_panel(ax, SCHEME_TITLES[scheme])
            continue
        for i, lab in enumerate(present):
            fam = "level" if lab == "AF" else "topology"
            _dots_and_mean(ax, i, sd.loc[sd["feature"] == lab, "score"].to_numpy(), FAMILY_COLORS[fam], rng=rng)
        ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=40, ha="right")
        ax.set_title(SCHEME_TITLES[scheme])
    for ax in axes[:, 0]:
        ax.set_ylabel(SCORE_LABEL)
    _family_legend(fig, ["topology", "level"])
    fig.suptitle(f"Level control (AF), topology and both, {grid}, {thr}", y=1.03, color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 03: grid normalization
# ---------------------------------------------------------------------------

GRID_A, GRID_B = "trial-mean grid", "min-max grid"


def grid_deltas(pp: pd.DataFrame) -> pd.DataFrame:
    """Per problem: trial-mean grid - min-max grid, for identically defined feature sets."""
    if pp.empty or not {GRID_A, GRID_B} <= set(pp["grid"]):
        return pd.DataFrame()
    keys = ["label", "thr", "family", "target", "scope", "unit_id"]
    a = pp[pp["grid"] == GRID_A]
    b = pp[pp["grid"] == GRID_B]
    # Moments are defined differently on the two grids (CV vs std), so they are not paired.
    a = a[~a["feature"].str.startswith("moments")]
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if m.empty:
        return m
    m["delta"] = m["score_a"] - m["score_b"]
    return m[keys + ["delta", "score_a", "score_b"]]


def plot_grid_comparison(gd: pd.DataFrame) -> Optional[plt.Figure]:
    if gd.empty:
        print("No feature set has results on both grids yet")
        return None
    order = feature_order(gd)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    rng = np.random.default_rng(0)
    for ax, scheme in zip(axes.ravel(), SCHEMES):
        sd = gd[(gd["target"] == scheme[0]) & (gd["scope"] == scheme[1])]
        labs = [l for l in order if l in set(sd["label"])]
        if not labs:
            _empty_panel(ax, SCHEME_TITLES[scheme])
            continue
        for i, lab in enumerate(labs):
            rows = sd[sd["label"] == lab]
            _dots_and_mean(ax, i, rows["delta"].to_numpy(), FAMILY_COLORS[rows["family"].iloc[0]], rng=rng)
        ax.axhline(0, color=INK_2, lw=0.9, zorder=1)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=50, ha="right")
        ax.set_title(SCHEME_TITLES[scheme])
    for ax in axes[:, 0]:
        ax.set_ylabel("trial-mean − min-max")
    _family_legend(fig, [f for f in ["3D-CNN", "topology", "level"] if f in set(gd["family"])])
    fig.suptitle(f"Effect of grid normalization: difference in {SCORE_LABEL} (> 0: trial-mean grid better)",
                 y=1.03, color=INK)
    fig.tight_layout()
    return fig
