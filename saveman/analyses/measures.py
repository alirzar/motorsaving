"""Manifold eccentricity vs functional connectivity measures."""

from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from brainspace.gradient.utils import dominant_set
import bct
from neuromaps import stats
from surfplot import Plot

from saveman.config import Config
from saveman.utils import get_surfaces, permute_map
from saveman.analyses import plotting

def compute_measures(cmat: pd.DataFrame, thresh: float = 0.9) -> pd.DataFrame:
    """Compute region-level connectivity measures (integration/segregation).

    Parameters
    ----------
    cmat
        Dense connectivity matrix as a DataFrame, indexed and columned by ROI
        names (same order). Values are typically covariance or correlation.
    thresh
        Row-wise threshold parameter used to create a sparse matrix via
        `brainspace.gradient.utils.dominant_set`. Default 0.9, matching
        thresholds used elsewhere in the pipeline.

    Returns
    -------
    pd.DataFrame
        One row per ROI with metadata columns (`roi`, `hemi`, `network`) and
        measures: strength, participation coefficients, and within-module
        degree z-score.
    """
    if not isinstance(cmat, pd.DataFrame):
        raise TypeError("`cmat` must be a pandas DataFrame with ROI labels as index/columns.")
    if cmat.shape[0] != cmat.shape[1]:
        raise ValueError(f"`cmat` must be square. Got shape {cmat.shape}.")
    if not (0 < thresh < 1):
        raise ValueError("`thresh` must be between 0 and 1 (exclusive).")

    # Parse ROI labels: expects something like "{something}_{hemi}_{network}_..."
    regions = cmat.index.to_series().str.split("_", n=3, expand=True)
    if regions.shape[1] < 3:
        raise ValueError(
            "ROI labels must contain at least 3 underscore-separated fields "
            "(e.g., '*_<hemi>_<network>_*')."
        )

    # Affiliation vectors (strings) for community definitions
    network_aff = regions[2].astype(str).to_numpy(copy=True)
    network_hemi_aff = (regions[1].astype(str) + "_" + regions[2].astype(str)).to_numpy(copy=True)

    # If your pipeline uses fixed blocks for cortex/subcortex/cerebellum, set them explicitly.
    # These assumptions should match your atlas ordering.
    if len(network_aff) >= 432:
        network_aff[400:432] = "Subcortex"
        network_aff[432:] = "Cerebellum"

        network_hemi_aff[400:416] = "LH_Subcortex"
        network_hemi_aff[416:432] = "RH_Subcortex"
        network_hemi_aff[432:448] = "LH_Cerebellum"
        network_hemi_aff[448:] = "RH_Cerebellum"

    # Convert community labels to integer community assignments for BCT functions.
    comm_net, _ = pd.factorize(network_aff, sort=True)
    comm_net_hemi, _ = pd.factorize(network_hemi_aff, sort=True)

    # Threshold and binarize
    x = dominant_set(cmat.values, k=1 - thresh, is_thresh=False, as_sparse=False)
    x_bin = (x != 0).astype(float)

    measures = pd.DataFrame(
        {
            "roi": cmat.index.values,
            "hemi": regions[1].astype(str).values,
            "network": network_aff,
            "participation": bct.participation_coef(x, comm_net, "out"),
            "participation_h": bct.participation_coef(x, comm_net_hemi, "out"),
            "module_degree": bct.module_degree_zscore(x_bin, comm_net, 2),
            "module_degree_h": bct.module_degree_zscore(x_bin, comm_net_hemi, 2),
            "strength": np.sum(x, axis=1),
        }
    )
    return measures


def correlate_measures(
    data: pd.DataFrame,
    out_dir: str,
    parc,
    n_perm: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Correlate connectivity measures with eccentricity using spin testing.

    Significance is determined via spin permutation testing: for each measure
    x, we correlate x with eccentricity y, then compare against correlations
    obtained with spun versions of x.

    Parameters
    ----------
    data
        Region-level DataFrame including a `distance` (eccentricity) column and
        one or more measure columns.
    out_dir
        Directory to save correlation results and permutation distributions.
    parc
        Parcellation passed through to `permute_map`. In this codebase, it is
        typically a (lh_gii, rh_gii) tuple.
    n_perm
        Number of spatial permutations to generate (default 1000).

    Returns
    -------
    correlations, distributions, spin_data
        correlations: DataFrame with columns [measure, r, p]
        distributions: DataFrame of null correlation distributions (one column per measure)
        spin_data: dict mapping measure -> spun maps (n_perm x n_regions)
    """
    if "distance" not in data.columns:
        raise ValueError("`data` must include a 'distance' column (eccentricity).")

    y = data["distance"].to_numpy()

    # Important: exclude metadata and the target variable itself.
    exclude = {"roi", "hemi", "network", "distance"}
    measures = [c for c in data.columns if c not in exclude]
    if len(measures) == 0:
        raise ValueError("No measure columns found to correlate (after excluding metadata and 'distance').")

    os.makedirs(out_dir, exist_ok=True)

    correlations = []
    distributions: Dict[str, np.ndarray] = {}
    spin_data: Dict[str, np.ndarray] = {}

    for m in measures:
        x = data[m].to_numpy()
        spins = permute_map(x, parcellation=parc, n_perm=n_perm)

        r = stats.efficient_pearsonr(x, y, nan_policy="omit")[0]
        nulls = stats.efficient_pearsonr(y, spins, nan_policy="omit")[0]
        p = (np.sum(np.abs(nulls) >= np.abs(r)) + 1) / (len(nulls) + 1)

        correlations.append({"measure": m, "r": float(r), "p": float(p)})
        distributions[m] = nulls
        spin_data[m] = spins

    correlations_df = pd.DataFrame(correlations)
    distributions_df = pd.DataFrame(distributions)

    # Save results
    correlations_df.to_csv(os.path.join(out_dir, "ecc_fc_corrs.tsv"), sep="\t", index=False)
    distributions_df.to_csv(os.path.join(out_dir, "ecc_fc_nulls.tsv"), sep="\t", index=False)

    for m, spins in spin_data.items():
        np.savetxt(os.path.join(out_dir, f"{m}_spins.tsv"), spins, delimiter="\t")

    return correlations_df, distributions_df, spin_data


def run_correlations(
    data: pd.DataFrame,
    out_dir: str,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Run correlations or load existing results from disk."""
    config = Config()
    lh_gii = os.path.join(config.resources, "lh_relabeled.gii")
    rh_gii = os.path.join(config.resources, "rh_relabeled.gii")

    if overwrite or (not os.path.exists(out_dir)):
        corrs, nulls, spins = correlate_measures(data, out_dir, parc=(lh_gii, rh_gii))
    else:
        corrs = pd.read_table(os.path.join(out_dir, "ecc_fc_corrs.tsv"))
        nulls = pd.read_table(os.path.join(out_dir, "ecc_fc_nulls.tsv"))

        spins = {}
        for fn in os.listdir(out_dir):
            if fn.endswith("_spins.tsv"):
                name = fn.replace("_spins.tsv", "")
                spins[name] = np.loadtxt(os.path.join(out_dir, fn))

    return corrs, nulls, spins


# ----------------------------
# Plotting helpers
# ----------------------------

def _plot_measure_map(
    data: pd.Series,
    atlas,
    cmap: str = "viridis",
    color_range: Optional[Tuple[float, float]] = None,
    cbar_kws: Optional[dict] = None,
):
    """Create a cortical surface map for a region-wise measure."""
    surfs = get_surfaces()
    x = plotting.weights_to_vertices(data, atlas)

    p = Plot(surfs["lh"], surfs["rh"])
    p.add_layer(x, cmap=cmap, zero_transparent=False, color_range=color_range)
    fig = p.build(cbar_kws=cbar_kws)
    return fig


def plot_maps(data: pd.DataFrame, out_dir: str) -> None:
    """Plot selected measures on cortical surface."""
    os.makedirs(out_dir, exist_ok=True)
    config = Config()

    # Common colorbar style
    cbar_kws = dict(location="right", n_ticks=3, aspect=7, shrink=0.15, draw_border=False, pad=-0.05)

    # Node strength
    fig = _plot_measure_map(data["strength"], config.atlas, cbar_kws=cbar_kws)
    fig.axes[0].set_title("Node strength", fontsize=14)
    fig.savefig(os.path.join(out_dir, "strength_map"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Participation coefficient
    fig = _plot_measure_map(
        data["participation"],
        config.atlas,
        cbar_kws=cbar_kws,
        color_range=(0, float(data["participation"].max())),
    )
    fig.axes[0].set_title("Participation\ncoefficient", fontsize=14)
    fig.savefig(os.path.join(out_dir, "participation_map"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Within-module degree z-score
    vmax = float(data["module_degree"].abs().max())
    fig = _plot_measure_map(
        data["module_degree"],
        config.atlas,
        color_range=(-vmax, vmax),
        cbar_kws=cbar_kws,
    )
    fig.axes[0].set_title("Within-module\ndegree z-score", fontsize=14)
    fig.savefig(os.path.join(out_dir, "module_degree_map"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(data: pd.DataFrame, out_dir: str) -> None:
    """Scatterplots of each measure vs eccentricity."""
    os.makedirs(out_dir, exist_ok=True)

    measures = ["strength", "participation", "module_degree"]
    xlabels = ["Node strength", "Participation\ncoefficient", "Within-module\ndegree z-score"]

    cmap = plotting.yeo_cmap(networks=9)

    for i, m in enumerate(measures):
        if m not in data.columns:
            continue

        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        sns.scatterplot(
            data=data,
            x=m,
            y="distance",
            ax=ax,
            hue="network",
            s=6,
            alpha=1,
            clip_on=False,
            palette=cmap,
            legend=False,
        )
        sns.regplot(x=m, y="distance", data=data, scatter=False, line_kws=dict(color="k", linewidth=1), ci=None, ax=ax)

        ax.set(xlabel=xlabels[i], ylabel="Eccentricity")
        sns.despine()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{m}_scatter"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    """Example entrypoint matching the original script behavior."""
    config = Config()

    # Create output directory for figures
    fig_dir = os.path.join(config.figures, "measures")
    os.makedirs(fig_dir, exist_ok=True)

    # Apply lab plotting style only when running as a script
    plotting.set_plotting()

    # Load connectivity matrix and eccentricity
    fname = os.path.join(config.connect + "-centered", "reference_cmat.tsv")
    cmat = pd.read_table(fname, index_col=0)

    data = compute_measures(cmat)

    ecc = pd.read_table(os.path.join(config.results, "ref_ecc.tsv"), sep="\t")
    data = data.merge(ecc, on="roi")

    data_cortex = data.query('network not in ["Subcortex", "Cerebellum"]').copy()

    out_dir = os.path.join(config.results, "fc_correlations")
    corrs, nulls, spins = run_correlations(data_cortex, out_dir, overwrite=True)

    plot_maps(data_cortex, fig_dir)
    plot_correlations(data, fig_dir)

    data.to_csv(os.path.join(out_dir, "fc_connectivity_measures.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
