"""Reference-gradient visualization and reference-manifold summary plots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import cmasher as cmr
from surfplot import Plot

from saveman.utils import load_table, load_gradients, get_surfaces, parse_roi_names
from saveman.config import Config
from saveman.analyses import plotting


def _get_config(config: Optional[Config] = None) -> Config:
    """Return a Config instance, using the provided one if given."""
    return config if config is not None else Config()

def _get_fig_dir(config: Optional[Config] = None, fig_dir: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve (and create) the figure output directory."""
    cfg = _get_config(config)
    out = Path(fig_dir) if fig_dir is not None else Path(cfg.figures) / "reference"
    out.mkdir(parents=True, exist_ok=True)
    return out


def show_eigenvalues(config: Optional[Config] = None) -> pd.DataFrame:
    """Load reference eigenvalues (variance explained) and add a cumulative column."""
    cfg = _get_config(config)
    fname = os.path.join(cfg.gradients, "reference_eigenvalues.tsv")
    df = load_table(fname)
    df["cumulative"] = df["proportion"].cumsum()
    return df


def plot_eigenvalues(config: Optional[Config] = None, k: int = 10) -> plt.Figure:
    """Plot variance explained and cumulative variance explained for the first k PCs."""
    cfg = _get_config(config)
    fname = os.path.join(cfg.results, "ref_eigenvalues.tsv")

    ev = pd.read_table(fname)
    eigenvals = ev["proportion"].iloc[:k]
    cum_eigenvals = ev["cumulative"].iloc[:k]

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.plot(np.arange(k) + 1, cum_eigenvals, marker="o", markerfacecolor="k", color="k")
    ax.grid(True)
    ax.bar(np.arange(k) + 1, eigenvals, zorder=2)

    ax.set_xlabel("Principal Component (PC)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variance Explained (proportion)", fontsize=12, fontweight="bold")
    ax.set(xticks=np.arange(0, k + 1, 2))
    sns.despine()
    fig.tight_layout(w_pad=3)
    return fig


def plot_ref_brain_gradients(
    k: int = 3,
    config: Optional[Config] = None,
    fig_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Plot region gradient weights on the cortical surface (cortex-only visualization)."""
    cfg = _get_config(config)
    out_dir = _get_fig_dir(cfg, fig_dir)

    fname = os.path.join(cfg.gradients, "reference_gradient.tsv")
    filtered_networks = ["Subcortex", "Cerebellum"]
    gradients = load_gradients(fname, k).query("network not in @filtered_networks")

    prefix = out_dir / "gradients_"
    cmap = cmr.get_sub_cmap("twilight_shifted", 0.05, 0.95)

    grads = gradients.filter(like="g").values
    vmax = float(np.around(np.nanmax(grads), decimals=1))

    # colorbar
    plotting.plot_cbar(cmap, -vmax, vmax, "horizontal", size=(1, 0.3), n_ticks=3)
    plt.savefig(str(prefix) + "cbar", bbox_inches="tight")
    plt.close()

    surfaces = get_surfaces()
    for i in range(k):
        x = plotting.weights_to_vertices(grads[:, i], cfg.atlas)
        p = Plot(surfaces["lh"], surfaces["rh"])
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax), cbar=False)
        fig = p.build(colorbar=False)
        fig.savefig(str(prefix) + f"PC{i + 1}_brain", bbox_inches="tight")
        plt.close(fig)


def plot_loading_distributions(
    k: int = 3,
    view_3d: Tuple[int, int] = (30, -120),
    config: Optional[Config] = None,
) -> Union[sns.axisgrid.JointGrid, plt.Figure]:
    """Plot loading distributions (2D jointplot / 3D scatter + KDE panels) by network."""
    cfg = _get_config(config)
    fname = os.path.join(cfg.gradients, "reference_gradient.tsv")
    df = load_gradients(fname, k)

    cmap = plotting.yeo_cmap(networks=9)

    if k == 2:
        g = sns.jointplot(
            x="g1",
            y="g2",
            hue="network",
            data=df,
            palette=cmap,
            legend=False,
            height=4.5,
            marginal_kws=dict(alpha=0.7),
            joint_kws=dict(linewidth=0, s=15),
        )
        g.ax_joint.set(xlabel="PC1", ylabel="PC2")
        return g

    if k == 3:
        df["c"] = df["network"].apply(lambda x: cmap[x])
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(nrows=16, ncols=16)

        ax1 = fig.add_subplot(gs[:, :7], projection="3d")
        plotting.plot_3d(df["g1"], df["g2"], df["g3"], color=df["c"], ax=ax1, view_3d=view_3d, s=15, lw=0.5, alpha=0.8)
        ax1.set(zlim=(-2, 2), ylim=(-2, 3))
        sns.set(style="darkgrid")

        ax2 = fig.add_subplot(gs[:8, 7:11])
        sns.scatterplot(x="g1", y="g2", hue="network", data=df, palette=cmap, legend=False, ax=ax2, linewidths=0, s=15, edgecolor="none")
        ax2.set(xlabel="", xticklabels=[], xlim=(-2, 3), ylim=(-2, 3))
        ax2.set_ylabel("PC2", fontsize=12, fontweight="bold")

        ax3 = fig.add_subplot(gs[8:, 7:11])
        sns.scatterplot(x="g1", y="g3", hue="network", data=df, palette=cmap, legend=False, ax=ax3, linewidths=0, s=15, edgecolor="none")
        ax3.set(xlim=(-2, 3), ylim=(-2, 3))
        ax3.set_xlabel("PC1", fontsize=12, fontweight="bold")
        ax3.set_ylabel("PC3", fontsize=12, fontweight="bold")

        for i, gg in zip([2, 7, 12], ["g1", "g2", "g3"]):
            ax = fig.add_subplot(gs[i : i + 3, 12:])
            sns.kdeplot(x=gg, hue="network", data=df, palette=cmap, fill=True, ax=ax, legend=False, alpha=0.6)
            ax.set(xlabel="", ylim=(0, 0.5), yticks=(0, 0.5), ylabel="", xlim=(-3, 4), xticks=range(-3, 5, 1), yticklabels=("0", "0.5"))
            if gg == "g3":
                ax.set_xlabel("Loading", fontsize=12, fontweight="bold")
            else:
                ax.set_xticklabels([])
            num = gg[1]
            ax.set_title(f"PC{num}", loc="right", y=0.5, fontsize=12, fontweight="bold")
            sns.despine()

        return fig

    if k == 4:
        fig, ax = plt.subplots(figsize=(3, 1.5))
        sns.kdeplot(x="g4", hue="network", data=df, palette=cmap, fill=True, ax=ax, legend=False, alpha=0.7)
        ymax = 0.20
        ax.set(xlabel="Loading", ylim=(0, ymax), yticks=(0, ymax), ylabel="", yticklabels=("0", f"{ymax}"))  # type: ignore[arg-type]
        sns.despine()
        return fig

    raise ValueError("k must be 2, 3, or 4 for these plots.")


def plot_eccentricity_calc(
    view_3d: Tuple[int, int] = (30, -120),
    config: Optional[Config] = None,
    fig_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Plot example eccentricity vectors for a few ROIs in the reference manifold."""
    cfg = _get_config(config)
    out_dir = _get_fig_dir(cfg, fig_dir)

    fname = os.path.join(cfg.gradients, "reference_gradient.tsv")
    df = load_gradients(fname, 3)

    cmap = plotting.yeo_cmap(networks=9)
    df["c"] = df["network"].apply(lambda x: cmap[x])

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection="3d")
    plotting.plot_3d(df["g1"], df["g2"], df["g3"], color=df["c"], ax=ax1, view_3d=view_3d, s=20, lw=0.5, alpha=0.5)

    centroid = np.array([0, 0, 0])
    for i in [87, 196, 455]:
        data = np.vstack([centroid, df.loc[i, ["g1", "g2", "g3"]].values])
        ax1.plot(data[:, 0], data[:, 1], data[:, 2], c="k", ls="--", linewidth=2)
        plotting.plot_3d(data[1, 0], data[1, 1], data[1, 2], color=df.loc[i, "c"], ax=ax1, alpha=1, edgecolor="k", s=40, view_3d=view_3d, zorder=600)
        ax1.scatter([0], [0], [0], color="k", marker="s", s=100, alpha=1, zorder=600)

    ax1.set(zlim=(-2, 2), ylim=(-2, 3))
    fig.savefig(out_dir / "ecc_calculation", bbox_inches="tight")
    plt.close(fig)


def reference_eccentricity(
    k: int = 3,
    view_3d: Tuple[int, int] = (30, -120),
    config: Optional[Config] = None,
    fig_dir: Optional[Union[str, os.PathLike]] = None,
) -> pd.DataFrame:
    """Compute reference eccentricity (distance to manifold centroid) and plot scatter + surface map."""
    cfg = _get_config(config)
    out_dir = _get_fig_dir(cfg, fig_dir)

    fname = os.path.join(cfg.gradients, "reference_gradient.tsv")
    df = load_gradients(fname, k)
    grads = df.filter(like="g").values

    centroid = np.mean(grads, axis=0)
    ecc = np.linalg.norm(grads - centroid, axis=1)

    filtered_networks = ["Subcortex", "Cerebellum"]
    ecc_cortex = ecc[df.query("network not in @filtered_networks").index]

    vmax = float(np.nanmax(ecc))
    vmin = float(np.nanmin(ecc))
    cmap = "viridis"

    prefix = out_dir / "ref_ecc_"

    if k == 2:
        fig, ax = plt.subplots()
        sc = ax.scatter(df["g1"].values, df["g2"].values, c=ecc, cmap=cmap, s=20, linewidths=0.2)
        ax.set(xlabel="PC1", ylabel="PC2", xlim=(-3, 4), ylim=(-3, 4))
        sns.despine()
        fig.savefig(str(prefix) + "scatter", bbox_inches="tight")
        plt.close(fig)

    elif k == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        plotting.plot_3d(grads[:, 0], grads[:, 1], grads[:, 2], c=ecc, ax=ax, cmap=cmap, lw=0.5, s=20, view_3d=view_3d)
        ax.set(zlim=(-2, 2), ylim=(-2, 3))
        fig.savefig(str(prefix) + "scatter", bbox_inches="tight")
        plt.close(fig)

    surfaces = get_surfaces()
    x = plotting.weights_to_vertices(ecc_cortex, cfg.atlas)

    p = Plot(surfaces["lh"], surfaces["rh"])
    p.add_layer(x, color_range=(vmin, vmax), cmap=cmap)
    cbar_kws = dict(location="bottom", decimals=2, fontsize=12, n_ticks=2, shrink=0.4, aspect=4, draw_border=False, pad=-0.06)
    fig = p.build(cbar_kws=cbar_kws)
    fig.savefig(str(prefix) + "brain", bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame({"roi": df["roi"], "distance": ecc})


def plot_ref_heatmaps(
    k: int = 3,
    config: Optional[Config] = None,
    fig_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Plot heatmaps of reference gradients and eccentricity grouped by network."""
    cfg = _get_config(config)
    out_dir = _get_fig_dir(cfg, fig_dir)

    fname = os.path.join(cfg.gradients, "reference_gradient.tsv")
    df = load_gradients(fname, k)
    df = parse_roi_names(df)

    grads = df.filter(like="g").set_index(df["network"])
    networks = grads.index.values

    net_names, starts, ends = [], [], []
    current = networks[0]
    start = 0
    for i, net in enumerate(networks[1:], 1):
        if net != current:
            net_names.append(current)
            starts.append(start)
            ends.append(i - 1)
            current = net
            start = i
    net_names.append(current)
    starts.append(start)
    ends.append(len(networks) - 1)

    fig, ax = plt.subplots(figsize=(4, 10))
    sns.heatmap(data=grads, ax=ax, cbar=True, cmap="RdBu_r", vmin=-2.9, vmax=2.9)
    ax.set(xticklabels=["PC1", "PC2", "PC3"], yticklabels=[], ylabel="", yticks=[])
    x_bracket, x_offset = -0.2, 0.05

    for name, s, e in zip(net_names, starts, ends):
        ax.plot([x_bracket, x_bracket], [s, e + x_offset], color="black", clip_on=False, linewidth=1)
        ax.plot([x_bracket, x_bracket + x_offset], [s, s - x_offset], color="black", clip_on=False, linewidth=1)
        ax.plot([x_bracket, x_bracket + x_offset], [e + x_offset, e + x_offset], color="black", clip_on=False, linewidth=1)
        mid = (s + e) / 2.0
        ax.text(x_bracket - 0.4, mid, name, ha="right", va="center", fontsize=12, rotation=0, clip_on=False)

    plt.tight_layout()
    fig.savefig(out_dir / "gradients_heatmap", bbox_inches="tight")
    plt.close(fig)

    ref = parse_roi_names(pd.read_table(os.path.join(cfg.results, "ref_ecc.tsv"), sep="\t"))
    ref = ref.set_index(ref["network"])[["distance"]]

    fig, ax = plt.subplots(figsize=(1, 10))
    sns.heatmap(data=ref, ax=ax, cbar=True, cmap="viridis")
    ax.set(xticklabels=["Eccentricity"], yticklabels=[], ylabel="", yticks=[])

    for name, s, e in zip(net_names, starts, ends):
        ax.plot([x_bracket, x_bracket], [s, e + x_offset], color="black", clip_on=False, linewidth=1)
        ax.plot([x_bracket, x_bracket + x_offset], [s, s - x_offset], color="black", clip_on=False, linewidth=1)
        ax.plot([x_bracket, x_bracket + x_offset], [e + x_offset, e + x_offset], color="black", clip_on=False, linewidth=1)
        mid = (s + e) / 2.0
        ax.text(x_bracket - 0.4, mid, name, ha="right", va="center", fontsize=12, rotation=0, clip_on=False)

    plt.tight_layout()
    fig.savefig(out_dir / "ref_ecc_heatmap", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate reference outputs (tables + figures)."""
    cfg = Config()
    out_dir = _get_fig_dir(cfg)

    # set plotting style once for scripts, but not at import time
    plotting.set_plotting()

    df = show_eigenvalues(cfg)
    df.to_csv(os.path.join(cfg.results, "ref_eigenvalues.tsv"), sep="\t", index=False)

    fig = plot_eigenvalues(cfg)
    fig.savefig(out_dir / "var_explained", bbox_inches="tight")
    plt.close(fig)

    plot_ref_brain_gradients(cfg.k, cfg, out_dir)
    plot_eccentricity_calc(config=cfg, fig_dir=out_dir)

    ecc = reference_eccentricity(cfg.k, config=cfg, fig_dir=out_dir)
    ecc.to_csv(os.path.join(cfg.results, "ref_ecc.tsv"), sep="\t", index=False)

    fig2 = plot_loading_distributions(k=cfg.k, config=cfg)
    # jointplot returns a JointGrid; handle both
    if hasattr(fig2, "savefig"):
        fig2.savefig(out_dir / "ref_networks", bbox_inches="tight")
        plt.close(fig2)  # type: ignore[arg-type]
    else:
        fig2.figure.savefig(out_dir / "ref_networks", bbox_inches="tight")
        plt.close(fig2.figure)

    plot_ref_heatmaps(k=cfg.k, config=cfg, fig_dir=out_dir)

if __name__ == "__main__":
    main()