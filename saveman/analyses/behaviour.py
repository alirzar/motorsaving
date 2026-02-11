"""Behaviour analyses for the motor adaptation task."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pingouin as pg
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from sklearn.decomposition import PCA

from neuromaps import nulls
from neuromaps.datasets import fetch_atlas
from neuromaps import stats  # for efficient_pearsonr (optional, used elsewhere)

from saveman.config import Config
from saveman.utils import get_surfaces, parse_roi_names
from saveman.analyses import plotting
from surfplot import Plot
import cmasher as cmr


# ----------------------------
# Small helpers
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _savefig(fig: plt.Figure, path_no_ext: str, dpi: int = 300) -> None:
    """Save a figure robustly as PNG and close it."""
    out = path_no_ext if path_no_ext.lower().endswith("") else (path_no_ext + "")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def convert_pvalue_to_asterisks(pvalue: float) -> str:
    if pvalue <= 0.0001:
        return "****"
    if pvalue <= 0.001:
        return "***"
    if pvalue <= 0.01:
        return "**"
    if pvalue <= 0.05:
        return "*"
    return "ns"


# ----------------------------
# Core plotting
# ----------------------------

def task_behaviour_plot(data: pd.DataFrame, fig_dir: str) -> None:
    """Plot group-average error throughout the task (3 styles).

    Parameters
    ----------
    data
        Trial-wise behavioural data including columns: sub, ses, trial_bin, error.
        Units should already be in degrees if you want degree-scaled plots.
    fig_dir
        Output directory for figures.
    """
    _ensure_dir(fig_dir)
    color = ["darkcyan", "#FFA066"]

    max_trial_bin = int(data["trial_bin"].max())
    num_sub = int(len(data["sub"].unique()))
    data3 = data.copy()
    data3["trial_bin"] = np.tile(np.arange(1, 2 * max_trial_bin + 1), num_sub)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(
        x="trial_bin",
        y="error",
        data=data3,
        hue="ses",
        palette=color,
        errorbar=("ci", 68),
        linewidth=1.5,
        ax=ax,
    )
    ax.set_yticks(np.arange(-20, 50, 10))
    ax.axhline(0, lw=1, c="k", ls="--")
    ax.set_xlabel("Trial Bin", fontsize=12, fontweight="bold")
    ax.set_ylabel("Angular error (°)", fontsize=12, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    _savefig(fig, os.path.join(fig_dir, "task_plot"))


def plot_behav_distribution(
    df: pd.DataFrame,
    fig_dir: str,
    name: str = "early",
    angle: str = "hitAngle_hand_good",
    auto_ticks: bool = False,
    seed: int = 1,
) -> None:
    """Show sample distribution (box + jittered points).

    Parameters
    ----------
    df
        DataFrame containing the column in `angle`.
    fig_dir
        Output directory for figures.
    name
        Label used for axis title and filename.
    angle
        Column name to plot.
    auto_ticks
        If True, suppress y-ticks.
    seed
        RNG seed for jitter reproducibility.
    """
    _ensure_dir(fig_dir)
    if angle not in df.columns:
        raise ValueError(f"Column '{angle}' not found in df.")

    plot_df = df.copy()
    plot_df["x"] = 1

    ymin = 15 * (np.nanmin(plot_df[angle]) // 15)
    ymax = 15 * (np.nanmax(plot_df[angle]) // 15)

    fig, ax = plt.subplots(figsize=(2, 4))
    box_line_color = "k"
    sns.boxplot(
        x="x",
        y=angle,
        data=plot_df,
        color="silver",
        boxprops=dict(edgecolor=box_line_color),
        medianprops=dict(color=box_line_color),
        whiskerprops=dict(color=box_line_color),
        capprops=dict(color=box_line_color),
        showfliers=False,
        width=0.5,
        ax=ax,
    )

    cmap = cmr.get_sub_cmap("RdYlGn", 0.05, 0.9)

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.01, 0.4, len(plot_df))
    ax.scatter(
        x=plot_df["x"] + jitter,
        y=plot_df[angle],
        c=plot_df[angle],
        ec="k",
        linewidths=1,
        cmap=cmap,
        clip_on=False,
    )

    if auto_ticks:
        ax.set(xticks=[])
    else:
        ax.set(xticks=[], yticks=np.arange(ymin, ymax + 15, 15))

    ax.set_xlabel(" ")
    ax.set_ylabel(name, fontsize=12, fontweight="bold")
    sns.despine(bottom=True)
    fig.tight_layout()
    _savefig(fig, os.path.join(fig_dir, f"{name}_error_distribution"))


def plot_early_error_change(df: pd.DataFrame, fig_dir: str) -> None:
    """Bar + paired lines for Day1 vs Day2 early error (paired test)."""
    _ensure_dir(fig_dir)
    data1 = df[["sub", "D1 early", "Savings"]].rename(columns={"D1 early": "error"})
    data1["day"] = 1
    data2 = df[["sub", "D2 early", "Savings"]].rename(columns={"D2 early": "error"})
    data2["day"] = 2
    data = pd.concat([data1, data2], axis=0)

    res = pg.pairwise_tests(data=data, dv="error", within="day", subject="sub")
    pval = float(res["p-unc"].values[0])
    pval_asterisks = convert_pvalue_to_asterisks(pval)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(
        data=data,
        x="day",
        y="error",
        hue="day",
        palette=["#80ffea", "#FFA066"],
        ax=ax,
        fill=True,
        alpha=0.4,
        lw=2,
    )
    sns.lineplot(
        data=data,
        x="day",
        y="error",
        units="sub",
        estimator=None,
        color="k",
        alpha=0.5,
        ax=ax,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Day1 Early", "Day2 Early"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Angular error", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_title(pval_asterisks, fontsize=16, fontweight="bold")
    sns.despine()
    for bar, c in zip(ax.patches, ["darkcyan", "#FFA066"]):
        bar.set_edgecolor(c)
        bar.set_linewidth(2)

    fig.tight_layout()
    _savefig(fig, os.path.join(fig_dir, "task_Day1Day2_early_error"))


# ----------------------------
# Eccentricity-behaviour correlation maps
# ----------------------------

def correlation_map(
    score: pd.DataFrame,
    gradients: pd.DataFrame,
    name: str,
    fig_dir: str,
    sig_style: Optional[str] = None,
    lateral_only: bool = False,
    angle: str = "hitAngle_hand_good",
    plot_maps: bool = True,
) -> pd.DataFrame:
    """Correlate a behavioural score with region eccentricity and plot a surface map.

    Parameters
    ----------
    score
        DataFrame with columns ['sub', <angle>], where <angle> is the score column.
    gradients
        DataFrame with columns ['sub', 'roi', 'roi_ix', 'distance', ...].
    name
        Prefix for outputs.
    fig_dir
        Output directory for maps.
    sig_style
        None (default): plot full r-map with FDR outline.
        'uncorrected': outline uncorrected p<.05.
        'corrected': show only FDR-significant r values.
    lateral_only
        If True, plot only lateral view for the main map panel.
    angle
        Column in `score` to correlate against.
    plot_maps
        If False, compute correlations but skip plotting.

    Returns
    -------
    pd.DataFrame
        ROI-wise correlation results with columns ['r', 'p', 'p_fdr'] indexed by ROI.
    """
    _ensure_dir(fig_dir)

    if angle not in score.columns:
        raise ValueError(f"Column '{angle}' not found in score DataFrame.")

    data = (
        gradients[["sub", "roi", "roi_ix", "distance"]]
        .pivot(index="sub", columns="roi", values="distance")
    )
    # preserve original ROI order
    data = data[gradients["roi"].unique().tolist()]

    # defensive alignment
    if not np.array_equal(score["sub"].values, data.index.values):
        raise ValueError("Subject ordering mismatch between score and gradients pivot.")

    res = data.apply(lambda x: pearsonr(score[angle], x), axis=0)
    res = res.T.rename(columns={0: "r", 1: "p"})
    _, res["p_fdr"] = pg.multicomp(res["p"].values, method="fdr_bh")

    if not plot_maps:
        return res

    # cortex-only (first 400 ROIs in this project’s ordering)
    res_cortex = res.iloc[:400, :].copy()

    config = Config()
    rvals = plotting.weights_to_vertices(res_cortex["r"], config.atlas)
    p_unc = plotting.weights_to_vertices(res_cortex["p"], config.atlas)
    p_unc = np.where(p_unc < 0.05, 1, 0)
    p_fdr = plotting.weights_to_vertices(res_cortex["p_fdr"], config.atlas)
    qvals = np.where(p_fdr < 0.05, 1, 0)

    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    sulc_params = dict(data=sulc, cmap="gray", cbar=False)

    vmax = float(np.nanmax(np.abs(res_cortex["r"])))

    cmap = ListedColormap(
        np.genfromtxt(os.path.join(config.resources, "colormap.csv"), delimiter=",")
    )

    if lateral_only:
        p1 = Plot(surfaces["lh"], surfaces["rh"], views="lateral", layout="column", size=(250, 350), zoom=1.5)
    else:
        p1 = Plot(surfaces["lh"], surfaces["rh"])
    p2 = Plot(surfaces["lh"], surfaces["rh"], views="dorsal", size=(150, 200), zoom=3.3)
    p3 = Plot(surfaces["lh"], surfaces["rh"], views="posterior", size=(150, 200), zoom=3.3)

    for p, suffix in zip([p1, p2, p3], ["", "_dorsal", "_posterior"]):
        p.add_layer(**sulc_params)
        cbar = True if suffix == "_dorsal" else False

        if sig_style is None:
            p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
            p.add_layer((np.nan_to_num(rvals * qvals) != 0).astype(float), cbar=False, as_outline=True, cmap="viridis")
        elif sig_style == "uncorrected":
            p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
            p.add_layer((np.nan_to_num(rvals * p_unc) != 0).astype(float), cbar=False, as_outline=True, cmap="binary")
        elif sig_style == "corrected":
            x = rvals * qvals
            vmin = float(np.nanmin(x[np.abs(x) > 0])) if np.any(np.abs(x) > 0) else -vmax
            p.add_layer(x, cbar=cbar, cmap=cmr.get_sub_cmap(cmap, 0.66, 1), color_range=(vmin, vmax))
            p.add_layer((np.nan_to_num(rvals * qvals) != 0).astype(float), cbar=False, as_outline=True, cmap="binary")
        else:
            raise ValueError("sig_style must be one of {None, 'uncorrected', 'corrected'}")

        if suffix == "_dorsal":
            cbar_kws = dict(location="bottom", decimals=2, fontsize=10, n_ticks=2, shrink=0.4, aspect=4, draw_border=False, pad=0.05)
            fig = p.build(cbar_kws=cbar_kws)
        else:
            fig = p.build()

        suffix_out = suffix + ("_corr" if sig_style is None else "")
        _savefig(fig, os.path.join(fig_dir, f"{name}_correlation_map{suffix_out}"))

    return res


def network_correlation_analysis(gradients: pd.DataFrame, score: pd.DataFrame, angle: str = "hitAngle_hand_good") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Network-level correlation between average eccentricity and behaviour."""
    data = gradients.copy()
    network_ecc = data.groupby(["sub", "network"])["distance"].mean().reset_index()
    data = network_ecc.merge(score, on="sub", how="left")

    res = data.groupby(["network"]).apply(lambda x: pg.corr(x["distance"], x[angle]), include_groups=False)
    res["p_fdr"] = pg.multicomp(res["p-val"].values, method="fdr_bh")[1]
    return res, data


def permute_maps(
    data: pd.DataFrame,
    parc,
    atlas: str = "fsLR",
    density: str = "32k",
    n_perm: int = 1000,
    seed: int = 1234,
    p_thresh: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform spin permutations on parcellated correlation data and aggregate by network."""
    config = Config()
    lh_gii = os.path.join(config.resources, "lh_relabeled.gii")
    rh_gii = os.path.join(config.resources, "rh_relabeled.gii")

    data = data.reset_index().rename(columns={"index": "roi"})
    data["roi_ix"] = np.arange(1, 465).astype(int)
    data = parse_roi_names(data)
    data = data.iloc[:400, :]  # cortex only

    surfaces = fetch_atlas(atlas, density)["sphere"]
    y = np.asarray(data["r"].values)

    spins = nulls.vasa(
        data=y,
        parcellation=(lh_gii, rh_gii),
        n_perm=n_perm,
        seed=seed,
        surfaces=surfaces,
    )

    spins_df = pd.concat([data, pd.DataFrame(spins)], axis=1)
    network_data = (
        spins_df.groupby(["network"]).agg("mean", numeric_only=True).reset_index().drop(columns="roi_ix")
    )

    nulls_dist = network_data.iloc[:, -n_perm:].values
    rvals = network_data["r"].values
    pvals = np.array([np.mean(np.abs(nulls_dist[i, :]) >= np.abs(rvals[i])) for i in range(len(rvals))])
    p_adj = pg.multicomp(pvals, method="fdr_bh")[1]

    significant_networks = p_adj < p_thresh
    network_data.insert(4, "pspin", pvals)
    network_data.insert(5, "pspin_fdr", p_adj)
    network_data.insert(6, "sig", significant_networks.astype(int))
    return spins_df, network_data


def plot_behav_corr(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    prefix1: str,
    prefix2: str,
    out_dir: str,
    linecolor: str = "k",
    auto_xticks: bool = True,
    auto_yticks: bool = True,
) -> None:
    """Scatterplot correlation between two subject-level measures."""
    _ensure_dir(out_dir)
    data = pd.merge(data1, data2, on="sub", how="left")
    rval, pval = pearsonr(data.iloc[:, -2], data.iloc[:, -1])

    fig = sns.lmplot(
        x=data.columns[-2],
        y=data.columns[-1],
        data=data,
        scatter_kws={"color": "k", "clip_on": False},
        line_kws={"color": linecolor},
        facet_kws={"sharex": True},
        height=4,
        aspect=0.8,
    )
    if not auto_xticks:
        plt.xticks([0, 15, 30, 45])
    if not auto_yticks:
        plt.yticks([0, 15, 30, 45])

    plt.xlabel(prefix1, fontsize=12, fontweight="bold")
    plt.ylabel(prefix2, fontsize=12, fontweight="bold")
    plt.title(f"r = {rval: .2f}\np = {pval: .4f}", ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = os.path.join(out_dir, f"behavior_{prefix1}_{prefix2}_correlation")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Behavioural metrics
# ----------------------------

def behav_metrics(sub_behav: pd.DataFrame, num_bins: int = 6) -> pd.DataFrame:
    """Compute epoch-level metrics and summary behavioural scores.

    Returns a wide subject table with columns:
    D1/D2 early/late, washout, Savings, SavingsRelative, PC1/PC2, etc.
    """
    epochs = ["early", "late", "washout-early", "washout-late"]
    data = sub_behav.copy()
    data["epoch"] = ""

    mask = data["trial_bin"] <= num_bins
    data.loc[mask, "epoch"] = "base"

    mask = (15 < data["trial_bin"]) & (data["trial_bin"] <= 15 + num_bins)
    data.loc[mask, "epoch"] = "early"

    mask = (56 - num_bins <= data["trial_bin"]) & (data["trial_bin"] < 56)
    data.loc[mask, "epoch"] = "late"

    mask = (56 <= data["trial_bin"]) & (data["trial_bin"] < 56 + num_bins)
    data.loc[mask, "epoch"] = "washout-early"

    mask = (70 - num_bins < data["trial_bin"]) & (data["trial_bin"] <= 70)
    data.loc[mask, "epoch"] = "washout-late"

    res = (
        data.groupby(["sub", "epoch", "ses"])["error"]
        .mean()
        .reset_index()
        .query("epoch in @epochs")
    )

    res = (
        res.pivot(index="sub", columns=["epoch", "ses"], values="error")
        .set_axis(
            ["D1 early", "D2 early", "D1 late", "D2 late", "WO1 early", "WO2 early", "WO1 late", "WO2 late"],
            axis=1,
        )
    )

    res["Savings"] = res["D1 early"] - res["D2 early"]
    res["SavingsRelative"] = (((res["D1 early"].abs()) - (res["D2 early"].abs())) / (res["D1 early"].abs())) * 100

    pca = PCA(n_components=2)
    PCs = pca.fit_transform(res)
    res["PC1"] = -PCs[:, 0]
    res["PC2"] = PCs[:, 1]

    # optional transformations used in downstream plots
    res["WO1 early"] = res["WO1 early"].abs()
    res["WO2 early"] = res["WO2 early"].abs()

    return res.reset_index()


def add_recall_ratio(df: pd.DataFrame, rotation_deg: float = 45.0, eps: float = 1e-6, min_learned_deg: float = 5.0) -> pd.DataFrame:
    """Tsay-style recall ratio using adaptation magnitude rather than error."""
    e1 = df["D1 late"].abs()
    e2 = df["D2 early"].abs()

    adapt_d1_late = np.clip(rotation_deg - e1, 0.0, rotation_deg)
    adapt_d2_early = np.clip(rotation_deg - e2, 0.0, rotation_deg)

    df["Adapt_D1Late"] = adapt_d1_late
    df["Adapt_D2Early"] = adapt_d2_early

    rr = adapt_d2_early / (adapt_d1_late + eps)
    if min_learned_deg is not None:
        rr = rr.where(adapt_d1_late >= min_learned_deg, np.nan)

    df["RecallRatio"] = rr
    return df


def add_learner_groups(df: pd.DataFrame, metric: str = "FPC1") -> pd.DataFrame:
    """Add fast vs slow learner grouping by median split on a behavioural metric."""
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in df.columns")
    median_val = df[metric].median()
    df["LearnerGroup"] = np.where(df[metric] >= median_val, "fast", "slow")
    return df


def plot_learning_curves_by_group(sub_behav: pd.DataFrame, fig_dir: str, name: str = "task_plot_fast_vs_slow") -> None:
    """Plot group-average error across the task for fast vs slow learners."""
    _ensure_dir(fig_dir)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    data = sub_behav.query("ses == 'ses-01'")
    ax = axs[0]
    sns.lineplot(
        data=data,
        x="trial_bin",
        y="error",
        hue="LearnerGroup",
        errorbar=("ci", 68),
        linewidth=1.5,
        ax=ax,
    )
    ax.axhline(0, lw=1, c="k", ls="--")
    ax.set_xlabel("Trial bin", fontsize=12, fontweight="bold")
    ax.set_ylabel("Angular error (°)", fontsize=12, fontweight="bold")
    data = sub_behav.query("ses == 'ses-02'")
    ax = axs[1]
    sns.lineplot(
        data=data,
        x="trial_bin",
        y="error",
        hue="LearnerGroup",
        errorbar=("ci", 68),
        linewidth=1.5,
        ax=ax,
    )
    ax.axhline(0, lw=1, c="k", ls="--")
    ax.set_xlabel("Trial bin", fontsize=12, fontweight="bold")
    ax.set_ylabel("Angular error (°)", fontsize=12, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    _savefig(fig, os.path.join(fig_dir, name))

def plot_region_correlations(epoch_gradients, score, fig_dir, epoch, col='FPC1', suffix=''):
    """Plot scatterplot for exemplar regions

    Parameters
    ----------
    gradients : pd.DataFrame
        Subject-level gradients dataset with distance column
    error : _type_
        Subject-level median error data
    """
    # pre-determined regions
    rois = {
        'ses-01_early-corrected': ['7Networks_RH_Default_pCunPCC_5'], 
        'ses-02_early-corrected': ['7Networks_RH_Default_PFCdPFCm_3'],
        'ses-01_WOearly-corrected': ['7Networks_RH_Default_pCunPCC_5'],
        'ses-02_WOearly-corrected': ['7Networks_RH_Default_PFCdPFCm_3']
        }
        
    cmap = plotting.yeo_cmap(networks=7)
    roi = rois[epoch]
    df = epoch_gradients.query("roi in @roi")
    df = df.merge(score, left_on='sub', right_on='sub')
    df['roi'] = df['roi'].str.replace('7Networks_', '')
    g = sns.lmplot(x='distance', y=col, col='roi', data=df, hue='network',
                   scatter_kws={'clip_on': False}, palette=cmap, legend=False,
                   facet_kws={'sharex': False}, height=2.3, aspect=.8, )
    g.set_xlabels('Eccentricity')
    g.set_ylabels('FPCA score')
    g.set(ylim=(-3, 2), yticks=np.arange(-3, 3, 1))
    g.tight_layout()
    g.savefig(os.path.join(fig_dir, f'{epoch}_example_roi_correlations{suffix}'))

def plot_savings_by_group(df: pd.DataFrame, fig_dir: str, metric: str = "Savings") -> None:
    """Compare savings between fast and slow learners."""
    _ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(
        data=df,
        x="LearnerGroup",
        y=metric,
        palette=["red", "green"],
        hue="LearnerGroup",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(data=df, x="LearnerGroup", y=metric, color="k", alpha=0.7, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    _savefig(fig, os.path.join(fig_dir, f"{metric}_fast_vs_slow"))

def plot_permute_maps(data, out_dir, n_perm=1000, p_thresh=.05):
    """
    Plots permutation test results for brain network correlations.

    Args:
        data (pd.DataFrame): DataFrame containing neuroimaging stats. 
            Expected columns include:
            - 'network': Names of the brain networks (e.g., Yeo 7 networks).
            - 'r': Observed correlation values.
            - 'pspin_fdr': FDR-corrected p-values from permutation testing.
            - Last n_perm columns: Null distribution values for each network.
        out_dir (str): Path (including filename and extension) where the 
            resulting figure will be saved.
        n_perm (int, optional): Number of permutation columns to include from 
            the end of the DataFrame. Defaults to 1000.
        p_thresh (float, optional): Significance threshold for coloring observed 
            points. Points <= p_thresh are red; otherwise blue. Defaults to 0.05.

    Returns:
        None: The figure is saved to the specified directory.
    """
    rvals = data['r'].values
    pspin_fdr = data['pspin_fdr'].values
    
    # Scale and prepare null distribution data
    nulls = pd.DataFrame(1.8 * data.iloc[:, -n_perm:].values)
    nulls_dist = pd.concat([data.iloc[:, 0], nulls], axis=1)
    melted_data = pd.melt(pd.DataFrame(nulls_dist), id_vars='network')
    
    cmap = plotting.yeo_cmap(networks=7)
    fig = plt.figure(figsize=(5, 5))
    
    sns.boxplot(data=melted_data, x='network', y='value', whis=[0, 100],
                palette=cmap, hue='network', saturation=.8, width=.5)
    
    # Overlay real correlation values as points
    for i, r in enumerate(rvals):
        # Color based on FDR-corrected p-values
        color = 'red' if pspin_fdr[i] <= p_thresh else 'blue'  
        plt.plot(i, r, color=color, marker='o', markersize=5, zorder=5)
        
    plt.axhline(0, color='blue', linestyle='dashed', zorder=-1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xticks(range(len(data)), data['network'], rotation=90, 
               ha='center', fontsize=12, fontweight='bold')
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    sns.despine(bottom=True)
    plt.xlabel('')
    plt.ylabel('Correlation (r)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(out_dir)
# ----------------------------
# Script entry
# ----------------------------

def main() -> None:
    config = Config()

    # Apply lab plotting style only when running as a script
    plotting.set_plotting()

    fig_dir = os.path.join(config.figures, "behaviour")
    _ensure_dir(fig_dir)

    sub_behav = pd.read_csv(os.path.join(config.resources, "subject_behavior_bin.csv"))
    sub_behav["error"] = (sub_behav["error"] * 180) / np.pi  # radians -> degrees

    num_bins = 6
    df = behav_metrics(sub_behav, num_bins)

    fpca_score = pd.read_table(os.path.join(config.results, "fpca", "D1D2-17bases_angular_error_bin.tsv"))
    df = pd.merge(df, fpca_score, on="sub", how="left")

    df = add_recall_ratio(df)                 # uses D1 late, D2 early
    df = add_learner_groups(df, metric="FPC1")  # or 'Savings'

    sub_behav = pd.merge(sub_behav, df[["sub", "LearnerGroup"]], on="sub", how="left")

    df.to_csv(os.path.join(config.results, f"behav_bin-{num_bins}bins.csv"), index=False)

    task_behaviour_plot(sub_behav, fig_dir)

    df["D1D2 early"] = df[["D1 early", "D2 early"]].mean(axis=1)
    plot_early_error_change(df, fig_dir)

    gradients = pd.read_table(os.path.join(config.results, "subject_gradients.tsv"))

    # Baseline eccentricity per subject, per ROI
    day1base = gradients.query('epoch == "base" & ses == "ses-01"')[["sub", "roi", "distance"]]
    day2base = gradients.query('epoch == "base" & ses == "ses-02"')[["sub", "roi", "distance"]]

    # Behavioural distributions & relationships
    plot_learning_curves_by_group(sub_behav, fig_dir)
    plot_savings_by_group(df, fig_dir)

    for col in ["FPC1", "RecallRatio"]:
        plot_behav_distribution(df[["sub", col]], fig_dir, name=col, angle=col, auto_ticks=True)

    plot_behav_corr(df[["sub", "D1 early"]], df[["sub", "D2 early"]], "D1 early", "D2 early", fig_dir, auto_xticks=False, auto_yticks=False)

    plot_behav_corr(df[["sub", "WO1 early"]], df[["sub", "FPC1"]], "abs WO1 early", "FPC1", fig_dir, auto_xticks=False)
    plot_behav_corr(df[["sub", "WO2 early"]], df[["sub", "FPC1"]], "abs WO2 early", "FPC1", fig_dir, auto_xticks=False)
    plot_behav_corr(df[["sub", "WO1 early"]], df[["sub", "WO2 early"]], "abs WO1 early", "abs WO2 early", fig_dir, auto_xticks=False, auto_yticks=False)

    plot_behav_corr(df[["sub", "D1 early"]], df[["sub", "FPC1"]], "D1 early", "FPC1", fig_dir, auto_xticks=False)
    plot_behav_corr(df[["sub", "D2 early"]], df[["sub", "FPC1"]], "D2 early", "FPC1", fig_dir, auto_xticks=False)

    plot_behav_corr(df[["sub", "Savings"]], df[["sub", "FPC1"]], "Saving", "FPC1", fig_dir)
    plot_behav_corr(df[["sub", "Savings"]], df[["sub", "RecallRatio"]], "Saving", "RecallRatio", fig_dir)
    plot_behav_corr(df[["sub", "FPC1"]], df[["sub", "RecallRatio"]], "FPC1", "RecallRatio", fig_dir)

    # Eccentricity–behaviour correlation maps (baseline-corrected eccentricity)

    n_perm = 1000

    # Day 1 early
    ses = "ses-01"
    epoch_gradients = gradients.query('epoch == "early" & ses == @ses').copy()
    epoch_gradients = epoch_gradients.merge(day1base, on=["sub", "roi"], how="left", suffixes=("", "_base"))
    epoch_gradients["distance"] = epoch_gradients["distance"] - epoch_gradients["distance_base"]
    epoch_gradients.drop(columns=["distance_base"], inplace=True)

    score = df[["sub", "FPC1"]]
    res = correlation_map(score, epoch_gradients, f"{ses}_early-corrected_fpca-score", fig_dir, angle="FPC1", sig_style="uncorrected")
    res.reset_index().to_csv(os.path.join(config.results, f"{ses}_early-corrected_fpca-score_correlations.tsv"), index=False, sep="\t")

    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm)
    network_spins.to_csv(os.path.join(config.results, f"{ses}_early-corrected_fpca-score_spins.tsv"), index=False, sep="\t")
    plot_region_correlations(epoch_gradients, score, fig_dir, epoch='ses-01_early-corrected')
    prefix = os.path.join(fig_dir, f'{ses}_early-corrected_fpca-score_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)
    # Day 2 early
    ses = "ses-02"
    epoch_gradients = gradients.query('epoch == "early" & ses == @ses').copy()
    epoch_gradients = epoch_gradients.merge(day2base, on=["sub", "roi"], how="left", suffixes=("", "_base"))
    epoch_gradients["distance"] = epoch_gradients["distance"] - epoch_gradients["distance_base"]
    epoch_gradients.drop(columns=["distance_base"], inplace=True)

    res = correlation_map(score, epoch_gradients, f"{ses}_early-corrected_fpca-score", fig_dir, angle="FPC1", sig_style="uncorrected")
    res.reset_index().to_csv(os.path.join(config.results, f"{ses}_early-corrected_fpca-score_correlations.tsv"), index=False, sep="\t")

    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm)
    network_spins.to_csv(os.path.join(config.results, f"{ses}_early-corrected_fpca-score_spins.tsv"), index=False, sep="\t")
    plot_region_correlations(epoch_gradients, score, fig_dir, epoch='ses-02_early-corrected')
    prefix = os.path.join(fig_dir, f'{ses}_early-corrected_fpca-score_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)
    
    # Day 1 WOearly
    ses = "ses-01"
    epoch_gradients = gradients.query('epoch == "washout-early" & ses == @ses').copy()
    epoch_gradients = epoch_gradients.merge(day1base, on=["sub", "roi"], how="left", suffixes=("", "_base"))
    epoch_gradients["distance"] = epoch_gradients["distance"] - epoch_gradients["distance_base"]
    epoch_gradients.drop(columns=["distance_base"], inplace=True)

    score = df[["sub", "FPC1"]]
    res = correlation_map(score, epoch_gradients, f"{ses}_WOearly-corrected_fpca-score", fig_dir, angle="FPC1", sig_style="uncorrected")
    res.reset_index().to_csv(os.path.join(config.results, f"{ses}_WOearly-corrected_fpca-score_correlations.tsv"), index=False, sep="\t")

    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm)
    network_spins.to_csv(os.path.join(config.results, f"{ses}_WOearly-corrected_fpca-score_spins.tsv"), index=False, sep="\t")
    plot_region_correlations(epoch_gradients, score, fig_dir, epoch='ses-01_WOearly-corrected')
    prefix = os.path.join(fig_dir, f'{ses}_WOearly-corrected_fpca-score_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)
    # Day 2 WOearly
    ses = "ses-02"
    epoch_gradients = gradients.query('epoch == "washout-early" & ses == @ses').copy()
    epoch_gradients = epoch_gradients.merge(day2base, on=["sub", "roi"], how="left", suffixes=("", "_base"))
    epoch_gradients["distance"] = epoch_gradients["distance"] - epoch_gradients["distance_base"]
    epoch_gradients.drop(columns=["distance_base"], inplace=True)

    res = correlation_map(score, epoch_gradients, f"{ses}_WOearly-corrected_fpca-score", fig_dir, angle="FPC1", sig_style="uncorrected")
    res.reset_index().to_csv(os.path.join(config.results, f"{ses}_WOearly-corrected_fpca-score_correlations.tsv"), index=False, sep="\t")

    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm)
    network_spins.to_csv(os.path.join(config.results, f"{ses}_WOearly-corrected_fpca-score_spins.tsv"), index=False, sep="\t")
    plot_region_correlations(epoch_gradients, score, fig_dir, epoch='ses-02_WOearly-corrected')
    prefix = os.path.join(fig_dir, f'{ses}_WOearly-corrected_fpca-score_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)
    # Recall Ratio with baseline corrected eccentricity - D1 late
    ses = "ses-01"
    epoch_gradients = gradients.query('epoch == "late" & ses == @ses').copy()
    epoch_gradients = epoch_gradients.merge(day1base, on=["sub", "roi"], how="left", suffixes=("", "_base"))
    epoch_gradients["distance"] = epoch_gradients["distance"] - epoch_gradients["distance_base"]
    epoch_gradients.drop(columns=["distance_base"], inplace=True)

    score_rr = df[["sub", "RecallRatio"]]
    res = correlation_map(score_rr, epoch_gradients, f"{ses}_late-corrected_recall_ratio", fig_dir, angle="RecallRatio", sig_style="uncorrected")
    res.reset_index().to_csv(os.path.join(config.results, f"{ses}_late-corrected_recall_ratio_correlations.tsv"), index=False, sep="\t")

    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm)
    network_spins.to_csv(os.path.join(config.results, f"{ses}_late-corrected_recall_ratio_spins.tsv"), index=False, sep="\t")
    prefix = os.path.join(fig_dir, f'{ses}_late-corrected_recall_ratio_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)

if __name__ == "__main__":
    main()
