"""General utilities used throughout the project."""

from __future__ import annotations

import os
import glob
import subprocess
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import natsort
import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
from brainspace.mesh.mesh_io import read_surface
import bct
from neuromaps import images
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from saveman.config import Config

pjoin = os.path.join


def get_files(pattern: Union[str, List[str]], force_list: bool = False):
    """Extract files in natural sort order matching a glob pattern."""
    if isinstance(pattern, list):
        pattern = pjoin(*pattern)

    files = natsort.natsorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("Pattern could not detect file(s)")

    if (not force_list) and (len(files) == 1):
        return files[0]
    return files


def check_img(img):
    """Load an image if not already loaded."""
    return nib.load(img) if isinstance(img, str) else img


def display(msg: str) -> None:
    """Print a timestamped message."""
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


def parse_roi_names(x: pd.DataFrame, col: str = "roi") -> pd.DataFrame:
    """Parse ROI strings into hemisphere/network/name, using project conventions."""
    x = x.copy()

    if "roi_ix" in x.columns:
        roi_cols = ["hemi", "network", "name"]
        x[roi_cols] = x[col].str.split("_", n=3, expand=True).iloc[:, 1:]
        x.loc[(401 <= x["roi_ix"]) & (x["roi_ix"] <= 416), "hemi"] = "LH"
        x.loc[(417 <= x["roi_ix"]) & (x["roi_ix"] <= 432), "hemi"] = "RH"
        x.loc[(433 <= x["roi_ix"]) & (x["roi_ix"] <= 448), "hemi"] = "LH"
        x.loc[(449 <= x["roi_ix"]), "hemi"] = "RH"
        x.loc[(401 <= x["roi_ix"]) & (x["roi_ix"] <= 432), "network"] = "Subcortex"
        x.loc[(433 <= x["roi_ix"]), "network"] = "Cerebellum"
        x.loc[(401 <= x["roi_ix"]) & (x["roi_ix"] <= 432), "name"] = (
            x.loc[(401 <= x["roi_ix"]) & (x["roi_ix"] <= 432), "roi"].str.replace("-rh", "").str.replace("-lh", "")
        )
        x.loc[(433 <= x["roi_ix"]), "name"] = x.loc[(433 <= x["roi_ix"]), "roi"]
        return x

    if len(x) == 464:
        roi_cols = ["hemi", "network", "name"]
        x[roi_cols] = x[col].str.split("_", n=3, expand=True).iloc[:, 1:]
        x.loc[400:416, "hemi"] = "LH"
        x.loc[416:432, "hemi"] = "RH"
        x.loc[432:448, "hemi"] = "LH"
        x.loc[448:, "hemi"] = "RH"
        x.loc[400:432, "network"] = "Subcortex"
        x.loc[432:, "network"] = "Cerebellum"
        x.loc[400:432, "name"] = x.loc[400:432, "roi"].str.replace("-rh", "").str.replace("-lh", "")
        x.loc[432:, "name"] = x.loc[432:, "roi"]
        return x

    raise ValueError("No Info To Parse the ROI Names")


def load_gradients(fname: str, k: Optional[int] = None) -> pd.DataFrame:
    """Read a gradient file and parse ROI information."""
    df = pd.read_table(fname, index_col=0)
    if k is not None:
        df = df.iloc[:, :k]

    df = df.reset_index().rename(columns={"index": "roi"})
    df = parse_roi_names(df)
    return df


def load_table(fname: str) -> pd.DataFrame:
    return pd.read_table(fname, index_col=0)


def get_surfaces(style: str = "inflated", load: bool = True):
    """Fetch surface files for a given surface style."""
    config = Config()
    surf_path = os.path.join(config.resources, "surfaces")
    surfaces = get_files([surf_path, f"*.{style}_*"])

    if load:
        surfs = [read_surface(i) for i in surfaces]
        return dict(zip(["lh", "rh"], surfs))
    return surfaces


def schaefer1000_roi_ix():
    x = np.arange(1000) + 1
    return x[~np.isin(x, [533, 903])]


def schaefer400_roi_ix():
    return np.arange(400) + 1


def schaefer400tian_roi_ix():
    return np.arange(432) + 1


def schaefer400tian_nettekoven_roi_ix():
    return np.arange(464) + 1


def get_roi_ix464() -> pd.DataFrame:
    """Return a (roi, roi_ix) table for the 464-region atlas ordering.

    NOTE: The original code referenced ``Config.results`` as a class attribute.
    This version uses an instance (``Config().results``).
    """
    config = Config()
    gradients = pd.read_table(os.path.join(config.results, "subject_gradients.tsv"))
    base = np.unique(gradients["epoch"].values)[0]
    roi_ix = gradients.query('sub == "sub-01" & epoch == @base & ses == "ses-01"')[["roi", "roi_ix"]]
    return roi_ix


def fdr_correct(x: pd.DataFrame, colname: str = "p-unc") -> pd.DataFrame:
    """Apply Benjamini–Hochberg FDR correction across all rows."""
    corrected = pg.multicomp(x[colname].values, method="fdr_bh")
    x[["sig_corrected", "p_fdr"]] = np.array(corrected).T
    return x


def parcellation_adjacency(dlabel: str, lh_surf: str, rh_surf: str, min_vertices: int = 1) -> np.ndarray:
    """Compute a parcel adjacency matrix using wb_command."""
    tmp = "adjacency.pconn.nii"
    cmd = (
        f"wb_command -cifti-label-adjacency {dlabel} {tmp} "
        f"-left-surface {lh_surf} -right-surface {rh_surf}"
    )
    subprocess.run(cmd.split(), check=True)

    adj = (nib.load(tmp).get_fdata() >= min_vertices).astype(float)
    os.remove(tmp)

    assert np.array_equal(adj, adj.T)
    remove_ix = np.where(np.sum(np.abs(adj), axis=1) == 0)[0]
    if len(remove_ix) > 0:
        adj = np.delete(adj, remove_ix, axis=0)
        adj = np.delete(adj, remove_ix, axis=1)
    return adj


def get_clusters(data: pd.DataFrame, adjacency: pd.DataFrame, sort: bool = True, yuh: bool = False) -> pd.DataFrame:
    data_cortex = data.query("roi_ix <= 400")
    ix = data_cortex.query("sig_corrected == 1")["roi_ix"].values

    adjacency = adjacency.copy()
    adjacency.columns = adjacency.columns.astype(int)
    x = adjacency.loc[ix, ix].values
    assignments, sizes = bct.get_components(x)

    cluster_table = pd.DataFrame({"cluster": np.arange(len(sizes)) + 1, "size": sizes})

    res = data_cortex.copy()
    res.index = res["roi_ix"].values
    res["cluster"] = 0
    res.loc[ix, "cluster"] = assignments
    res = res.merge(cluster_table, on="cluster", how="left")
    res["size"] = np.nan_to_num(res["size"])

    if sort:
        labels = res.sort_values("size", ascending=False)["cluster"].unique()
        new_labels = np.concatenate([np.arange(len(labels[:-1])) + 1, [0]])
        relabel_map = dict(zip(labels, new_labels))
        res["cluster"] = res["cluster"].apply(lambda v: relabel_map[v])

    data = data.copy()
    data["cluster"] = 0
    data["size"] = 0
    res = pd.concat([res, data.query("roi_ix > 400")])
    return res


def test_regions(data: pd.DataFrame, method: str = "anova", factor: str = "epoch", p_thresh: float = 0.05) -> pd.DataFrame:
    test = dict(anova=pg.rm_anova, ttest=pg.pairwise_tests)
    if method not in test:
        raise ValueError(f"method must be one of {list(test.keys())}")

    test_data = data[["sub", "roi", "roi_ix", "epoch", "ses", "distance"]]
    kwargs = dict(correction=True) if method == "anova" else {}

    if factor == "epoch * ses":
        res = {}
        for epoch in ["base", "early", "late", "washout-early", "washout-late"]:
            time_test_data = test_data.query("epoch == @epoch")
            res[epoch] = (
                time_test_data.groupby(["roi", "roi_ix"], sort=False)
                .apply(test[method], dv="distance", within="ses", subject="sub", include_groups=False, **kwargs)
                .reset_index()
                .drop("level_2", axis=1)
            )
            res[epoch]["Contrast"] = "epoch * ses"
            res[epoch].insert(3, "epoch", epoch, allow_duplicates=False)
        res = pd.concat([res[i] for i in res], ignore_index=True).sort_values(by="roi_ix").reset_index()
    else:
        res = (
            test_data.groupby(["roi", "roi_ix"], sort=False)
            .apply(test[method], dv="distance", within=factor, subject="sub", include_groups=False, **kwargs)
            .reset_index()
            .drop("level_2", axis=1)
        )

    res["sig"] = (res["p-unc"] < p_thresh).astype(float)
    res = fdr_correct(res)
    return res


def permute_map(
    data,
    atlas: str = "fslr",
    density: str = "32k",
    parcellation=None,
    n_perm: int = 1000,
    surfaces=None,
    spins=None,
    seed: int = 1234,
):
    """Perform spin permutations on parcellated data.

    This wraps ``neuromaps.nulls.cornblath`` for parcel-wise vectors.
    """
    from neuromaps.datasets import fetch_atlas
    from neuromaps import nulls

    if parcellation is None:
        raise ValueError(
            "Cannot use `cornblath()` null method without specifying a parcellation."
        )
    y = np.asarray(data)
    if surfaces is None:
        surfaces = fetch_atlas(atlas, density)["sphere"]
    return nulls.cornblath(y, atlas, density, parcellation, n_perm=n_perm, spins=spins, seed=seed, surfaces=surfaces)


def optimal_k(x: np.ndarray, out_dir: str, prefix: str, show: bool = False) -> List[int]:
    """Find candidate optimal k for k-means via elbow + silhouette analysis.

    Parameters
    ----------
    x
        Feature matrix (n_samples × n_features).
    out_dir
        Directory where the diagnostic figure will be saved.
    prefix
        Filename prefix for the figure.
    show
        If True, display the figure. Default False (publication-friendly).

    Returns
    -------
    list[int]
        Candidate k values at local maxima of the silhouette score.
    """
    os.makedirs(out_dir, exist_ok=True)

    range_n_clusters = list(range(2, 12))
    elbow = []
    silhouette = []

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=1234)
        preds = clusterer.fit_predict(x)
        elbow.append(clusterer.inertia_)
        silhouette.append(silhouette_score(x, preds))

    k_opt: List[int] = []
    for i in range(1, len(silhouette) - 1):
        if silhouette[i - 1] < silhouette[i] > silhouette[i + 1]:
            k_opt.append(i + 2)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS (Inertia)", color="tab:red")
    ax1.plot(range_n_clusters, elbow, color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:blue")
    ax2.plot(range_n_clusters, silhouette, color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    ax1.set_title("Elbow and Silhouette Analysis")
    ax1.grid(True)

    out_path = os.path.join(out_dir, f"{prefix}elbow_silhouette_analysis.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return k_opt
