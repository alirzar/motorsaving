"""Plotting utilities used throughout the project."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn.plotting import cm
import cmasher as cmr
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
from surfplot.utils import add_fslr_medial_wall

from saveman.config import Config
from saveman.utils import get_surfaces


# ----------------------------
# Styling
# ----------------------------

def set_plotting() -> None:
    """Set project-wide plotting defaults.

    Parameters
    ----------
    save_format
        If provided, sets ``plt.rcParams['savefig.format']`` (e.g., 'png' or 'svg').
        If None, the default matplotlib format is left unchanged.
    """
    plt.rcParams["font.family"] = ["Arial"]
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.format"] = "svg"
    sns.set_context("paper")


def save_prefix(prefix: str) -> str:
    """Append the configured figure directory to a filename prefix."""
    config = Config()
    return os.path.join(config.figures, prefix + "_")


# ----------------------------
# Surface mapping
# ----------------------------

def _align_labels_to_atlas(x: np.ndarray, source_labels: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
    """Match labels to corresponding vertex labels."""
    target = np.unique(target_labels)[1:]
    df1 = pd.DataFrame(target, index=target)
    df2 = pd.DataFrame(x, index=source_labels)
    return pd.concat([df1, df2], axis=1).iloc[:, 1:].values


def weights_to_vertices(
    data: Union[np.ndarray, str, pd.Series, pd.DataFrame],
    target: Union[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
) -> Union[np.ndarray, list]:
    """Map parcel-wise weights to surface vertices in an atlas label space.

    If ``labels`` is not specified, values in ``data`` are mapped to ``target`` in
    ascending order.

    Parameters
    ----------
    data
        Array containing region weights. If a string, must be a valid CIFTI file
        containing parcel-wise values.
    target
        CIFTI file name (dlabel/dscalar) or a label vector defining vertex-space.
    labels
        Numeric labels for each region (row of ``data``) as they appear in the
        atlas vertices. Required when ``data`` contains fewer regions than the
        full atlas (e.g., thresholded / significant-only outputs).

    Returns
    -------
    numpy.ndarray or list[numpy.ndarray]
        Vertex-mapped weights (or a list for multi-column inputs).
    """
    if isinstance(target, str):
        vertices = nib.load(target).get_fdata().ravel()
    else:
        vertices = np.asarray(target).ravel()

    x = np.asarray(data)
    if labels is not None:
        x = _align_labels_to_atlas(x, np.asarray(labels), vertices)

    mask = vertices != 0
    map_args = dict(target_lab=vertices, mask=mask, fill=np.nan)

    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        return map_to_labels(x.ravel(), **map_args)
    else:
        return [map_to_labels(col, **map_args) for col in x.T]


def get_sulc() -> np.ndarray:
    """Get sulcal depth map for plotting background."""
    config = Config()
    surf_path = os.path.join(config.resources, "surfaces")
    img = os.path.join(surf_path, "S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii")
    vertices = nib.load(img).get_fdata().ravel()
    return add_fslr_medial_wall(vertices)


# ----------------------------
# Colormaps / helpers
# ----------------------------

def yeo_cmap(as_palette: bool = False, networks: int = 7):
    """Color map for Yeo network parcellations."""
    if networks == 17:
        cmap = {
            "VisCent": (120, 18, 136),
            "VisPeri": (255, 0, 2),
            "SomMotA": (70, 130, 181),
            "SomMotB": (43, 204, 165),
            "DorsAttnA": (74, 156, 61),
            "DorsAttnB": (0, 118, 17),
            "SalVentAttnA": (196, 58, 251),
            "SalVentAttnB": (255, 153, 214),
            "TempPar": (9, 41, 250),
            "ContA": (230, 148, 36),
            "ContB": (136, 50, 75),
            "ContC": (119, 140, 179),
            "DefaultA": (255, 254, 1),
            "DefaultB": (205, 62, 81),
            "DefaultC": (0, 0, 132),
            "LimbicA": (224, 248, 166),
            "LimbicB": (126, 135, 55),
        }
    elif networks == 19:
        cmap = {
            "VisCent": (120, 18, 136),
            "VisPeri": (255, 0, 2),
            "SomMotA": (70, 130, 181),
            "SomMotB": (43, 204, 165),
            "DorsAttnA": (74, 156, 61),
            "DorsAttnB": (0, 118, 17),
            "SalVentAttnA": (196, 58, 251),
            "SalVentAttnB": (255, 153, 214),
            "TempPar": (9, 41, 250),
            "ContA": (230, 148, 36),
            "ContB": (136, 50, 75),
            "ContC": (119, 140, 179),
            "DefaultA": (255, 254, 1),
            "DefaultB": (205, 62, 81),
            "DefaultC": (0, 0, 132),
            "LimbicA": (224, 248, 166),
            "LimbicB": (126, 135, 55),
            "Subcortex": (0, 0, 0),
            "Cerebellum": (100, 100, 100),
        }
    elif networks == 7:
        cmap = {
            "Vis": (119, 20, 140),
            "SomMot": (70, 126, 175),
            "DorsAttn": (0, 117, 7),
            "SalVentAttn": (195, 59, 255),
            "Limbic": (219, 249, 165),
            "Cont": (230, 149, 33),
            "Default": (205, 65, 80),
        }
    elif networks == 9:
        cmap = {
            "Vis": (119, 20, 140),
            "SomMot": (70, 126, 175),
            "DorsAttn": (0, 117, 7),
            "SalVentAttn": (195, 59, 255),
            "Limbic": (219, 249, 165),
            "Cont": (230, 149, 33),
            "Default": (205, 65, 80),
            "Subcortex": (0, 0, 0),
            "Cerebellum": (100, 100, 100),
        }
    else:
        raise ValueError("`networks` must be one of {7, 9, 17, 19}")

    cmap = {k: np.array(v) / 255 for k, v in cmap.items()}
    return sns.color_palette(list(cmap.values())) if as_palette else cmap


def plot_cbar(
    cmap,
    vmin: float,
    vmax: float,
    orientation: str = "vertical",
    size: Optional[Tuple[float, float]] = None,
    n_ticks: int = 2,
    decimals: int = 2,
    fontsize: int = 12,
    show_outline: bool = False,
    as_int: bool = False,
):
    """Plot a standalone colorbar and return the colorbar object."""
    if size is None and orientation == "vertical":
        size = (0.3, 4)
    if size is None and orientation == "horizontal":
        size = (4, 0.3)

    x = np.array([[0, 1]])
    plt.figure(figsize=size)
    plt.imshow(x, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])

    ticks = np.linspace(0, 1, n_ticks)
    tick_labels = np.around(np.linspace(vmin, vmax, n_ticks), decimals)
    cbar = plt.colorbar(orientation=orientation, cax=cax, ticks=ticks)
    cbar.set_ticklabels(tick_labels.astype(int) if as_int else tick_labels)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if not show_outline:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)
    return cbar


def plot_3d(x, y, z, ax=None, view_3d=(30, -120), **kwargs):
    """Plot a 3D scatter plot of region loadings/weights."""
    if ax is None:
        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(projection="3d")

    ax.scatter(xs=x, ys=y, zs=z, **kwargs)
    ax.set_xlabel("PC1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=12, fontweight="bold")
    ax.set_zlabel("PC3", fontsize=12, fontweight="bold")
    ax.set(xticks=np.arange(-2, 4), yticks=np.arange(-2, 4), zticks=np.arange(-2, 4))

    if view_3d is not None:
        ax.view_init(elev=view_3d[0], azim=view_3d[1])
    return ax


def stat_cmap():
    return cmr.get_sub_cmap(cm.cyan_orange, 0, 1)


def epoch_order():
    return {
        "rest": 1,
        "leftbaseline": 2,
        "rightbaseline": 3,
        "rightlearning-early": 4,
        "rightlearning-late": 5,
        "lefttransfer-early": 6,
        "lefttransfer-late": 7,
        "baseline": 1,
        "early": 2,
        "late": 3,
        "left": 1,
        "right": 2,
    }


def pairwise_stat_maps(
    data: pd.DataFrame,
    data_posthoc: pd.DataFrame,
    prefix: str = "",
    layercbar: bool = False,
    dorsal: bool = True,
    posterior: bool = True,
    vmax: Union[str, float] = "auto",
    vmin: Union[str, float] = "auto",
    cbar_orientation: str = "vertical",
    sig_style: str = "corrected",
):
    """Plot pairwise comparisons as t-maps on brain surfaces.

    Notes
    -----
    The original version computed ``tvals_all`` only for ``sig_style is None``
    but then used it for auto-scaling in all cases. This version computes a
    full t-value matrix for scaling regardless of ``sig_style``.
    """
    # Full t-values for scaling (all regions, no significance filter)
    all_contrasts = []
    for (A, B), g in data.groupby(["A", "B"]):
        tmp = g.set_index("roi_ix")[["T"]].rename(columns={"T": f"{B}_{A}_T"})
        all_contrasts.append(tmp)
    df_all = pd.concat(all_contrasts, axis=1)
    tvals_all = -df_all.filter(like="_T")  # flip sign: condition B is positive

    # Significant-only t-values (posthoc)
    sig_contrasts = []
    for (A, B), g in data_posthoc.groupby(["A", "B"]):
        sig = g[g["sig_corrected"].astype(bool)].set_index("roi_ix")[["T"]].rename(columns={"T": f"{B}_{A}_T"})
        sig_contrasts.append(sig)
    df_sig = pd.concat(sig_contrasts, axis=1) if sig_contrasts else pd.DataFrame()
    tvals = -df_sig.filter(like="_T") if not df_sig.empty else pd.DataFrame()

    # Auto scaling
    if vmax == "auto":
        vmax = float(np.nanmax(tvals_all.values))
    if vmin == "auto":
        vmin = float(np.nanmin(np.abs(tvals_all.values)))

    size = (0.2, 1) if cbar_orientation == "vertical" else (0.8, 0.25)

    cmap = cmr.get_sub_cmap("RdBu_r", 0.05, 0.95) if sig_style == "corrected" else cmr.get_sub_cmap("seismic", 0.05, 0.95)

    pos_cmap = cmr.get_sub_cmap(cmap, 0.51, 1)
    pos_cmap = cmr.get_sub_cmap(pos_cmap, vmin / vmax, 1)
    plot_cbar(pos_cmap, vmin, vmax, cbar_orientation, size=size, n_ticks=2)
    plt.savefig(prefix + "cbar_pos.png", dpi=300, bbox_inches="tight")
    plt.close()

    neg_cmap = cmr.get_sub_cmap(cmap, 0, 0.5)
    neg_cmap = cmr.get_sub_cmap(neg_cmap, 0, 1 - vmin / vmax)
    plot_cbar(neg_cmap, -vmax, -vmin, cbar_orientation, size=size, n_ticks=2)
    plt.savefig(prefix + "cbar_neg.png", dpi=300, bbox_inches="tight")
    plt.close()

    config = Config()
    surfaces = get_surfaces()
    sulc = get_sulc()

    sulc_params = dict(data=sulc, cmap="gray", cbar=False)
    layer_params = dict(cmap=cmap, cbar=layercbar, color_range=(-vmax, vmax))
    outline_params = dict(cbar=False, cmap="binary", as_outline=True)

    # Choose which t-values to plot
    plot_matrix = tvals_all if sig_style is None else tvals

    for col in plot_matrix.columns:
        contrast = col[:-2].replace("_", "_vs_")  # strip "_T"
        roi_ix = plot_matrix.index.values
        x = weights_to_vertices(plot_matrix[col].values, config.atlas, roi_ix)

        p = Plot(surfaces["lh"], surfaces["rh"], layout="row", mirror_views=True, size=(800, 200), zoom=1.2)
        p.add_layer(**sulc_params)
        p.add_layer(x, **layer_params)

        if sig_style == "corrected":
            p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
        fig = p.build()
        fig.savefig(prefix + contrast + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        if dorsal:
            p = Plot(surfaces["lh"], surfaces["rh"], views="dorsal", size=(150, 200), zoom=3.3)
            p.add_layer(**sulc_params)
            p.add_layer(x, cmap=cmap, cbar=False, color_range=(-vmax, vmax))
            if sig_style == "corrected":
                p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
            fig = p.build()
            fig.savefig(prefix + contrast + "_dorsal.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        if posterior:
            p = Plot(surfaces["lh"], surfaces["rh"], views="posterior", size=(150, 200), zoom=3.3)
            p.add_layer(**sulc_params)
            p.add_layer(x, **layer_params)
            if sig_style == "corrected":
                p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
            fig = p.build(colorbar=False)
            fig.savefig(prefix + contrast + "_posterior.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    return True


def plot_timeseries(fname: str):
    """Plot an exemplar timeseries for a methods figure."""
    df = pd.read_table(fname)
    df = df.iloc[:, [49, 144, 480]]
    cmap = yeo_cmap()

    scale_factor = 7
    fig, ax = plt.subplots(figsize=(5, 1))
    for i, roi in enumerate(["Default", "SomMot", "Vis"]):
        const = i * scale_factor if i == 0 else (i + 1) * scale_factor
        ax.plot(df.iloc[:, i] + const, c=cmap[roi], lw=1)

    ax.set_axis_off()
    fig.tight_layout()
    return fig
