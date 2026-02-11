"""Seed-based connectivity analysis across task epochs."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import pingouin as pg
import cmasher as cmr
from surfplot import Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from saveman.config import Config
from saveman.utils import get_files, get_roi_ix464, fdr_correct, parse_roi_names
from saveman.analyses import plotting
from saveman.analyses.plotting import get_sulc, get_surfaces, weights_to_vertices


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _savefig(fig: plt.Figure, path_no_ext: str, dpi: int = 300) -> None:
    out = path_no_ext if path_no_ext.lower().endswith("") else (path_no_ext + "")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def get_epoch_name(fname: str) -> str:
    """Extract epoch label from a connectivity filename."""
    epochs = ["base", "early", "late", "washout-early", "washout-late"]
    # try to find any epoch token in the filename
    match = None
    for e in epochs:
        if e in fname:
            match = e
            break
    if match is None:
        # fallback: strip extension
        return os.path.splitext(fname)[0]
    return match


def connect_seed(cmats: Sequence[str], seed_region: str, ref_rois: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Extract seed connectivity by isolating the seed row in each matrix."""
    if ref_rois is None:
        ref_rois = get_roi_ix464()

    rows = []
    for path in cmats:
        cmat = pd.read_table(path, index_col=0)
        if seed_region not in cmat.index:
            raise ValueError(f"Seed region '{seed_region}' not found in matrix index: {path}")

        res = pd.DataFrame(cmat.loc[seed_region].reset_index().values, columns=["roi", "r"])
        res["r"] = res["r"].astype(float)
        res = ref_rois.merge(res[["roi", "r"]], on="roi", how="left")

        fname = os.path.basename(path)
        parts = fname.split("_")
        res["sub"] = parts[0] if len(parts) > 0 else "unknown"
        res["ses"] = parts[1] if len(parts) > 1 else "unknown"
        res["epoch"] = get_epoch_name(fname)

        rows.append(res)

    connectivity = pd.concat(rows, ignore_index=True)
    return connectivity


def seed_analysis(cmats: Sequence[str], seed: str, epochs_all: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Perform seed connectivity contrasts across epoch pairs (paired tests)."""
    connectivity = connect_seed(cmats, seed)

    res_all = []
    for epochs_pair in epochs_all:
        df = connectivity.query("epoch in @epochs_pair").copy()
        if df.empty:
            continue

        res = (
            df.groupby(["roi", "roi_ix"], sort=False)
            .apply(pg.pairwise_tests, dv="r", within="epoch", subject="sub", include_groups=False)
            .reset_index()
            .drop(columns=["level_2"], errors="ignore")
        )

        # Swap sign so that B condition is the positive condition
        res["T"] = -res["T"]
        res["sig"] = (res["p-unc"] < 0.05).astype(float)
        res_all.append(res)

    if not res_all:
        raise ValueError("No results produced — check your epoch labels and input matrices.")

    res = pd.concat(res_all, axis=0, ignore_index=True)
    return fdr_correct(res)


def plot_seed_map(
    data: pd.DataFrame,
    seed_region: str,
    fig_dir: str,
    sig_style: Optional[str] = None,
    views: Sequence[str] = ("lateral", "medial"),
    use_fdr: bool = True,
    seed_color: str = "yellow",
    cortical_seed: bool = True,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Generate a seed-connectivity contrast surface map."""
    _ensure_dir(fig_dir)

    sig_regions = data.query("sig_corrected == 1") if use_fdr else data.query("sig == 1")

    config = Config()
    x = weights_to_vertices(data["T"].astype(float).values, config.atlas, data["roi_ix"].values)
    y = weights_to_vertices(np.ones(len(sig_regions)), config.atlas, sig_regions["roi_ix"].values)

    if cortical_seed:
        seed_mask = (data["roi"] == seed_region).astype(float)
        z = weights_to_vertices(seed_mask.values, config.atlas, data["roi_ix"].values)
        seed_cmap = LinearSegmentedColormap.from_list("regions", [seed_color, "k"], N=2)

    surfs = get_surfaces()
    sulc = get_sulc()
    p = Plot(surfs["lh"], surfs["rh"], views=list(views))
    p.add_layer(data=sulc, cmap="gray", cbar=False)

    if vmax is None:
        vmax = float(np.nanmax(np.abs(x))) if np.any(np.isfinite(x)) else 1.0
        suffix = seed_region
    else:
        suffix = "all_seeds"

    cmap = cmr.get_sub_cmap("seismic", 0.15, 0.85)

    # Save a colorbar image (useful for figures)
    plotting.plot_cbar(cmap, -vmax, vmax, "horizontal", size=(1, 0.3), n_ticks=3)
    plt.savefig(os.path.join(fig_dir, f"cbar_{suffix}"), dpi=300, bbox_inches="tight")
    plt.close()

    if sig_style == "trace":
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))
        p.add_layer(np.nan_to_num(y), as_outline=True, cmap="binary", cbar=False)
    elif sig_style == "threshold":
        p.add_layer(x * np.nan_to_num(y), cmap=cmap, color_range=(-vmax, vmax))
    else:
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))

    if cortical_seed:
        p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
        p.add_layer(np.nan_to_num(z), as_outline=True, cmap="binary", cbar=False)

    fig = p.build(colorbar=False)
    return fig


def get_seed_color() -> Dict[str, Dict[str, float]]:
    return {
        "early_vs_base": dict(color="darkblue", alpha=0.15),
        "late_vs_early": dict(color="brown", alpha=0.15),
        "washout-early_vs_base": dict(color="green", alpha=0.15),
        "washout-late_vs_washout-early": dict(color="purple", alpha=0.10),
    }


def plot_spider(res_list: Sequence[pd.DataFrame], out_path_no_ext: str) -> plt.Figure:
    """Spider plot of network-averaged T values across contrasts."""
    colors = get_seed_color()
    custom_order = list(plotting.yeo_cmap(networks=9).keys())

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    all_values: List[float] = []

    for res, label in zip(res_list, colors.keys()):
        data = parse_roi_names(res.copy())
        data_net = data.groupby(["network"]).mean(numeric_only=True).reset_index()
        data_net["network"] = pd.Categorical(data_net["network"], categories=custom_order, ordered=True)
        data_net = data_net.sort_values("network")

        values_net = data_net["T"].tolist()
        all_values += values_net
        network_labels = data_net["network"].tolist()

        num_vars = len(values_net)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        values_net = values_net + values_net[:1]
        angles = angles + angles[:1]

        cmap = colors[label]
        ax.fill(angles, values_net, **cmap, zorder=6)
        ax.plot(angles, values_net, linestyle="-", color=cmap["color"], zorder=5, linewidth=3)

    circle_angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(circle_angles, [0] * 100, linestyle="-", color="k", linewidth=3)

    label_angles = np.degrees(np.linspace(0, 2 * np.pi, len(network_labels), endpoint=False))
    ax.set_thetagrids(label_angles, network_labels, fontsize=14, fontweight="bold", zorder=8)
    ax.set_rlabel_position(10)

    ax.set_facecolor("#e0e0e0")
    ax.xaxis.grid(color="white", linestyle="-", linewidth=3, alpha=0.9)
    ax.yaxis.grid(color="white", linestyle="-", linewidth=3, alpha=0.9)

    if all_values:
        tmin = float(np.floor(min(all_values)))
        tmax = float(np.ceil(max(all_values)))
        ax.set_rgrids(np.arange(tmin, tmax + 1, 1), fontsize=12, fontweight="bold")

    _savefig(fig, out_path_no_ext)
    return fig


def plot_spider_legend(out_dir: str) -> None:
    """Save a standalone legend for the spider plot."""
    _ensure_dir(out_dir)
    colors = [c["color"] for c in get_seed_color().values()]
    labels = [
        "Baseline → Early",
        "Early → Late",
        "Baseline → Washout-early",
        "Washout-early → Washout-late",
    ]
    patches = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", label=label)
        for color, label in zip(colors, labels)
    ]
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.legend(handles=patches, fontsize=12, ncol=1, edgecolor="w", handlelength=1.5, handleheight=1.5)
    ax.axis("off")
    _savefig(fig, os.path.join(out_dir, "spider_legend"))


def plot_seed_eccentricity(gradients: pd.DataFrame, seed: str, out_path_no_ext: str) -> None:
    """Plot seed eccentricity across epochs for all subjects."""
    df = gradients[gradients["roi"] == seed].copy()
    if df.empty:
        raise ValueError(f"No rows found for seed ROI '{seed}' in gradients table.")

    g = sns.FacetGrid(
        data=df,
        col_wrap=2,
        col="ses",
        hue="ses",
        palette=["darkcyan", "#FFA066"],
        height=2.5,
        sharey=True,
        aspect=0.75,
    )
    g.map_dataframe(sns.lineplot, x="epoch", y="distance", errorbar=None, marker="o", ms=6, lw=1.3, color="k")
    g.map_dataframe(sns.stripplot, x="epoch", y="distance", jitter=0.1, zorder=-1, s=4, alpha=0.7)
    g.set_axis_labels("", "Eccentricity")
    g.set_xticklabels(["Baseline", "Early", "Late", "Washout-early", "Washout-late"], rotation=90)
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(title.replace("ses = ", ""))

    out = out_path_no_ext if out_path_no_ext.lower().endswith("") else (out_path_no_ext + "")
    g.savefig(out, dpi=300)
    plt.close("all")


def main() -> None:
    config = Config()
    plotting.set_plotting()

    cmats = get_files(os.path.join(config.dataset_dir, "connectivity", "sub*/*.tsv"))

    gradients = pd.read_table(os.path.join(config.results, "subject_gradients.tsv"))

    res_dir = os.path.join(config.results, "seed")
    fig_dir = os.path.join(config.figures, "seed")
    _ensure_dir(res_dir)
    _ensure_dir(fig_dir)

    seeds = [
        "7Networks_LH_SomMot_26", "7Networks_RH_SomMot_31",
        "7Networks_RH_Default_PFCdPFCm_7",
        "7Networks_LH_Default_pCunPCC_5", "7Networks_RH_Default_pCunPCC_4",
        "7Networks_LH_Default_PFC_9",
        "7Networks_LH_Vis_27", "7Networks_RH_Vis_26",
    ]
    epochs_all = [["base", "early"], ["early", "late"], ["base", "washout-early"], ["washout-early", "washout-late"]]

    for seed in seeds:
        plot_seed_eccentricity(gradients, seed, os.path.join(fig_dir, f"{seed}_ecc"))

        res = seed_analysis(cmats, seed=seed, epochs_all=epochs_all)

        seed_results = []
        for epochs_pair in epochs_all:
            suffix = f"{epochs_pair[1]}_vs_{epochs_pair[0]}"
            res_epoch = res.query("A == @epochs_pair[0] & B == @epochs_pair[1]").copy()
            res_epoch.to_csv(os.path.join(res_dir, f"{suffix}_{seed}.tsv"), sep="\t", index=False)
            seed_results.append(res_epoch)

            res_cor = res_epoch[res_epoch["roi_ix"] <= 400]
            fig = plot_seed_map(res_cor, seed, fig_dir=fig_dir, sig_style=None, vmax=7, views=("lateral", "medial"))
            _savefig(fig, os.path.join(fig_dir, f"{suffix}_{seed}"))

            fig = plot_seed_map(res_cor, seed, fig_dir=fig_dir, sig_style=None, vmax=7, views=("dorsal",))
            _savefig(fig, os.path.join(fig_dir, f"{suffix}_{seed}_dorsal"))

            fig = plot_seed_map(res_cor, seed, fig_dir=fig_dir, sig_style=None, vmax=7, views=("posterior",))
            _savefig(fig, os.path.join(fig_dir, f"{suffix}_{seed}_posterior"))

        plot_spider(seed_results, os.path.join(fig_dir, f"{seed}_spider"))

    plot_spider_legend(fig_dir)

if __name__ == "__main__":
    main()
