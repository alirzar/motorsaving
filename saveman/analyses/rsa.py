"""Representational Similarity Analysis (RSA) across task epochs."""

from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import rankdata

import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
from surfplot import Plot

from saveman.config import Config
from saveman.analyses import plotting


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _savefig(fig: plt.Figure, path_no_ext: str, dpi: int = 300) -> None:
    out = path_no_ext if path_no_ext.lower().endswith("") else (path_no_ext + "")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _epoch_order() -> List[str]:
    """Define the chronological order of epoch×session labels."""
    return [
        "base_ses-01",
        "early_ses-01",
        "late_ses-01",
        "washout-early_ses-01",
        "washout-late_ses-01",
        "base_ses-02",
        "early_ses-02",
        "late_ses-02",
        "washout-early_ses-02",
        "washout-late_ses-02",
    ]


def custom_sort(group: pd.DataFrame) -> pd.DataFrame:
    """Sort a group by the canonical epoch order."""
    epoch_order = _epoch_order()
    group = group.copy()
    group["epoch"] = pd.Categorical(group["epoch"], categories=epoch_order, ordered=True)
    return group.sort_values(by=["epoch"])


# ----------------------------
# Mapping / visualization
# ----------------------------

def mean_stat_map(
    data: pd.DataFrame,
    out_dir: str,
    prefix: str = "",
    centering: Optional[str] = None,
) -> None:
    """Plot the mean eccentricity map per epoch on the cortical surface.

    Parameters
    ----------
    data
        Subject gradient data with columns at least:
        ['epoch', 'ses', 'roi', 'roi_ix', 'distance'].
    out_dir
        Figure output directory.
    prefix
        Filename prefix for outputs.
    centering
        - None: plot raw mean eccentricity per epoch.
        - 'mean': mean-center each ROI across epochs.
        - '<epoch_label>': subtract that epoch column from all other epochs
          (e.g., 'base_ses-01') and plot relative values (centering epoch is dropped).
    """
    _ensure_dir(out_dir)

    mean_ecc = (
        data.groupby(["epoch", "ses", "roi", "roi_ix"])["distance"]
        .mean()
        .reset_index()
        .sort_values(["epoch", "roi_ix"])
        .query("roi_ix <= 400")
    )

    # ROI × epoch matrix in chronological order for plotting
    epoch_ecc = pd.DataFrame(
        {name: g["distance"].values for name, g in mean_ecc.groupby(["epoch", "ses"])}
    )
    epoch_ecc.columns = [f"{e[0]}_{e[1]}" for e in epoch_ecc.columns]
    roi_ix = mean_ecc["roi_ix"].unique()

    if centering is not None:
        if centering == "mean":
            epoch_ecc = epoch_ecc.sub(epoch_ecc.mean(axis=1), axis=0)
            prefix2 = f"centered_mean_ecc_{prefix}_"
        else:
            if centering not in epoch_ecc.columns:
                raise ValueError(f"Centering epoch '{centering}' not found. Available: {list(epoch_ecc.columns)}")
            for e in epoch_ecc.columns:
                if e != centering:
                    epoch_ecc[e] = epoch_ecc[e] - epoch_ecc[centering]
            epoch_ecc = epoch_ecc.drop(columns=[centering])
            prefix2 = f"relative_mean_ecc_{prefix}_"

        cmap = sns.diverging_palette(250, 0, s=100, l=30, sep=10, as_cmap=True)
        vmax = float(np.nanmax(epoch_ecc.values))
        vmin = float(np.nanmin(epoch_ecc.values))
        n_ticks = 3
    else:
        cmap = "viridis"
        vmax = 3.6
        vmin = 1.0
        n_ticks = 2
        prefix2 = f"mean_ecc_{prefix}_"

    # Save a colorbar image
    plotting.plot_cbar(cmap, vmin, vmax, "horizontal", size=(2, 0.2), n_ticks=n_ticks)
    plt.savefig(os.path.join(out_dir, f"{prefix2}cbar"), dpi=300, bbox_inches="tight")
    plt.close()

    config = Config()
    surfaces = plotting.get_surfaces()

    for col in epoch_ecc.columns:
        x = plotting.weights_to_vertices(epoch_ecc[col].values, config.atlas, roi_ix)
        p = Plot(surfaces["lh"], surfaces["rh"], mirror_views=False, size=(400, 400), zoom=1.2)
        p.add_layer(x, cmap=cmap, color_range=(vmin, vmax), cbar=False)
        fig = p.build()
        _savefig(fig, os.path.join(out_dir, f"{prefix2}{col}_brain"))


# ----------------------------
# RSA computation
# ----------------------------

def rsa_obj_to_df(rsm_obj) -> pd.DataFrame:
    """Convert an rsatoolbox RDM object into a long-form DataFrame of RSM values."""
    subs = rsm_obj.rdm_descriptors["sub"]
    epochs = rsm_obj.pattern_descriptors["epoch"]
    epoch_pairs = list(combinations(epochs, 2))

    rows = []
    for i, sub in enumerate(subs):
        rdm_values = rsm_obj[i].dissimilarities.flatten()
        rsm_values = 1.0 - rdm_values
        rows.append(
            pd.DataFrame(
                {
                    "sub": sub,
                    "epoch_1": [e[0] for e in epoch_pairs],
                    "epoch_2": [e[1] for e in epoch_pairs],
                    "distance": rsm_values,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def rsa_analysis(
    data: pd.DataFrame,
    cols: str = "distance",
    method: str = "correlation",
):
    """Compute epoch-level RDMs over ROIs for each subject.

    Parameters
    ----------
    data
        Must contain ['sub', 'epoch', 'ses', cols], where `cols` is an ROI-wise value.
        The RSA patterns are defined as epoch×session, and channels are ROIs.
    cols
        Column name holding the ROI-wise values (default 'distance').
    method
        'correlation' (Pearson correlation distance) or 'spearman' (rank-transform then Pearson).
        Other rsatoolbox methods may work (e.g., 'euclidean').

    Returns
    -------
    rsatoolbox.rdm.RDMs
        Subject-wise RDMs across epoch patterns.
    """
    required = {"sub", "epoch", "ses", cols}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Build a matrix: (sub, epoch, ses) × ROI (channels)
    sub_data = pd.DataFrame(
        {name: g[cols].values for name, g in data.groupby(["sub", "epoch", "ses"])}
    ).T

    subs = data["sub"].unique()
    use_spearman = method.lower() == "spearman"
    method2 = "correlation" if use_spearman else method

    datasets = []
    for sub in subs:
        measurements = sub_data.loc[sub]  # (n_epochs x n_roi), index is (epoch, ses)
        meas_array = measurements.values

        if use_spearman:
            meas_array = np.apply_along_axis(rankdata, 1, meas_array)

        des = {"sub": sub}
        obs_des = {"epoch": [f"{i[0]}_{i[1]}" for i in measurements.index.values]}
        chn_des = {"region": measurements.columns}

        datasets.append(
            rsd.Dataset(
                measurements=meas_array,
                descriptors=des,
                obs_descriptors=obs_des,
                channel_descriptors=chn_des,
            )
        )

    return rsr.calc_rdm(datasets, descriptor="epoch", method=method2)


def analyze_epoch_distance(
    df: pd.DataFrame,
    anchor_epoch: str = "early_ses-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """Repeated-measures ANOVA and targeted post-hoc tests for an anchor epoch.

    This focuses on distances between `anchor_epoch` and all other non-baseline epochs.

    Returns
    -------
    anova_results, post_hoc_results, fig
    """
    epoch_order = [
        "late_ses-01",
        "washout-early_ses-01",
        "washout-late_ses-01",
        "early_ses-02",
        "late_ses-02",
        "washout-early_ses-02",
        "washout-late_ses-02",
    ]
    cmap = ["darkcyan" if "ses-01" in e else "#FFA066" for e in epoch_order]

    epoch_data = df.query(
        "(epoch_1 == @anchor_epoch or epoch_2 == @anchor_epoch) and "
        'epoch_1 not in ["base_ses-01", "base_ses-02"] and '
        'epoch_2 not in ["base_ses-01", "base_ses-02"]'
    ).reset_index(drop=True)

    if epoch_data.empty:
        raise ValueError(f"No rows match anchor_epoch='{anchor_epoch}' after filtering.")

    # Ensure anchor is always in epoch_1 (swap where needed)
    mask = epoch_data["epoch_2"] == anchor_epoch
    if mask.any():
        epoch_data.loc[mask, ["epoch_1", "epoch_2"]] = epoch_data.loc[mask, ["epoch_2", "epoch_1"]].values

    epoch_data["epoch_2"] = pd.Categorical(epoch_data["epoch_2"], categories=epoch_order, ordered=True)

    # RM ANOVA across epoch_2
    anova_results = pg.rm_anova(data=epoch_data, dv="distance", within="epoch_2", subject="sub")

    # Post-hoc: compare a chosen epoch (default early_ses-02) to each alternative epoch_2
    alt_epoch1 = "early_ses-02"
    alt_epochs = np.unique(epoch_data["epoch_2"].astype(str).values)
    alt_epochs = alt_epochs[alt_epochs != alt_epoch1]

    res = []
    for alt_epoch2 in alt_epochs:
        data2 = epoch_data.query("epoch_2 in [@alt_epoch1, @alt_epoch2]").copy()
        data2["epoch_2"] = pd.Categorical(data2["epoch_2"], categories=[alt_epoch1, alt_epoch2], ordered=True)
        res.append(
            pg.pairwise_tests(
                data=data2,
                dv="distance",
                within="epoch_2",
                subject="sub",
                alternative="greater",
            )
        )
    post_hoc_results = pd.concat(res, ignore_index=True) if res else pd.DataFrame()

    # Plot
    fig, ax = plt.subplots(figsize=(2.8, 4))
    sns.lineplot(
        data=epoch_data,
        x="epoch_2",
        y="distance",
        errorbar=None,
        marker="o",
        ms=8,
        lw=1.5,
        mfc="k",
        mec="k",
        color="k",
        ax=ax,
    )
    sns.stripplot(
        data=epoch_data,
        x="epoch_2",
        y="distance",
        size=8,
        alpha=0.7,
        hue="epoch_2",
        palette=cmap,
        legend=False,
        jitter=0.05,
        zorder=-1,
        ax=ax,
    )

    # Annotate significant comparisons
    if not post_hoc_results.empty:
        sig_results = post_hoc_results[post_hoc_results["p-unc"] < 0.05]
        epoch_to_x = {ep: i for i, ep in enumerate(epoch_order)}

        y_max = float(epoch_data["distance"].max())
        y_offset = 0.10 * y_max if y_max != 0 else 0.05

        for i, (epoch_a, epoch_b, p_value) in enumerate(sig_results[["A", "B", "p-unc"]].values):
            if epoch_a not in epoch_to_x or epoch_b not in epoch_to_x:
                continue
            x1, x2 = epoch_to_x[epoch_a], epoch_to_x[epoch_b]
            y_pos = y_max + (i + 1) * y_offset

            if p_value < 0.001:
                sig_label = "***"
            elif p_value < 0.01:
                sig_label = "**"
            elif p_value < 0.05:
                sig_label = "*"
            else:
                sig_label = "ns"

            ax.plot([x1, x2], [y_pos, y_pos], color="black", linewidth=1.5)
            ax.text((x1 + x2) / 2, y_pos - 0.04, sig_label, ha="center", va="bottom", fontsize=16)

    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_xticks(range(len(epoch_order)))
    ax.set_xticklabels(epoch_order, rotation=90)
    sns.despine()
    fig.tight_layout()

    return anova_results, post_hoc_results, fig


# ----------------------------
# Script entry
# ----------------------------

def main() -> None:
    config = Config()

    # Apply lab plotting style only when running as a script
    plotting.set_plotting()

    fig_dir = os.path.join(config.figures, "rsa")
    res_dir = os.path.join(config.results, "rsa")
    _ensure_dir(fig_dir)
    _ensure_dir(res_dir)

    gradients = pd.read_table(os.path.join(config.results, "subject_gradients.tsv"))
    anova = pd.read_table(os.path.join(config.results, "ecc_anova_stats_ensemble.tsv"))
    behav_info = pd.read_csv(os.path.join(config.results, "behav_bin-6bins.csv"))

    for method in ["correlation", "spearman"]:
        suffix = "pearson" if method == "correlation" else "spearman"

        # Whole-sample RSA per cluster
        for cl in range(1, 5):
            cluster_label = f"epoch_sig_regions_cluster{cl}_{suffix}"

            epoch_sig_regions = anova.query(
                'Source == "epoch" & sig_corrected == 1 & ensemble_4 == @cl'
            )["roi"].values

            epoch_data = gradients.query("roi in @epoch_sig_regions")

            # Mean map per epoch (raw)
            mean_stat_map(epoch_data, fig_dir, prefix=cluster_label)

            # RSA
            rdm_obj = rsa_analysis(epoch_data, method=method)
            res_df = rsa_obj_to_df(rdm_obj)

            # Epoch distance stats
            anova_res, post_hoc, fig = analyze_epoch_distance(res_df)
            anova_res.to_csv(os.path.join(res_dir, f"{cluster_label}_epoch_comparison_anova.tsv"), sep="\t", index=False)
            post_hoc.to_csv(os.path.join(res_dir, f"{cluster_label}_epoch_comparison_posthoc.tsv"), sep="\t", index=False)
            _savefig(fig, os.path.join(fig_dir, f"{cluster_label}_epoch_comparison"))

        # Group-wise RSA (fast vs slow)
        for gp in ["fast", "slow"]:
            gp_sub = behav_info.loc[behav_info["LearnerGroup"] == gp]["sub"].tolist()
            gp_gradients = gradients.loc[gradients["sub"].isin(gp_sub)]

            for cl in range(1, 5):
                cluster_label = f"epoch_sig_regions_cluster{cl}_{suffix}_{gp}"

                epoch_sig_regions = anova.query(
                    'Source == "epoch" & sig_corrected == 1 & ensemble_4 == @cl'
                )["roi"].values

                epoch_data = gp_gradients.query("roi in @epoch_sig_regions")

                rdm_obj = rsa_analysis(epoch_data, method=method)
                res_df = rsa_obj_to_df(rdm_obj)

                anova_res, post_hoc, fig = analyze_epoch_distance(res_df)
                anova_res.to_csv(os.path.join(res_dir, f"{cluster_label}_epoch_comparison_anova.tsv"), sep="\t", index=False)
                post_hoc.to_csv(os.path.join(res_dir, f"{cluster_label}_epoch_comparison_posthoc.tsv"), sep="\t", index=False)
                _savefig(fig, os.path.join(fig_dir, f"{cluster_label}_epoch_comparison"))


if __name__ == "__main__":
    main()
