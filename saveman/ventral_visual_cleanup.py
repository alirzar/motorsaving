"""Utilities to define ventral visual parcels from Schaefer atlas centroids and 
construct nuisance regressors (mean signal or PCA-derived component) that are removed 
from cerebellar time series via linear regression to mitigate ventral visual contamination."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Optional, Literal, Tuple, Dict, Sequence, Iterable
from matplotlib.colors import LinearSegmentedColormap

from surfplot import Plot
from saveman.config import Config

from saveman.utils import get_surfaces
from saveman.analyses import plotting

def load_schaefer_centroids(path: str,
                            name_col: str = "ROI Name",
                            x_col: str = "R",
                            y_col: str = "A",
                            z_col: str = "S") -> pd.DataFrame:
    """
    Load the Schaefer centroid table.

    The file should have at least:
      - a column with parcel names matching the time-series columns
      - x, y, z (MNI coordinates)

    Adjust `sep` and column names if needed.
    """
    df = pd.read_csv(path, sep=",")
    df = df[[name_col, x_col, y_col, z_col]].copy()
    df.columns = ["roi", "x", "y", "z"]
    return df

def get_ventral_visual_parcels(centroids: pd.DataFrame,
                               z_thresh: float = 0.0,
                               y_thresh: Optional[float] = None) -> List[str]:
    """
    Identify ventral visual parcels using Schaefer centroids.

    - Visual parcels: name contains 'Vis'
    - Ventral: z < z_thresh (default: z < 0)
    - Optional: posterior constraint y < y_thresh
    
    Returns a list of parcel names.
    """
    vis = centroids[centroids["roi"].str.contains("Vis")]
    mask = vis["z"] < z_thresh
    if y_thresh is not None:
        mask = mask & (vis["y"] < y_thresh)
    ventral = vis[mask]
    return ventral["roi"].tolist()

def build_visual_floor_regressors(
    ts: pd.DataFrame,
    ventral_visual_names: List[str],
    mode: Literal["mean", "pca"] = "mean",
    n_pcs: int = 3,
) -> np.ndarray:
    """
    Construct visual-floor regressors from the time-series DataFrame.

    ts:        T x 464 DataFrame (columns include Schaefer parcels)
    ventral_visual_names: list of column names for ventral visual parcels
    mode:      "mean" → single average regressor
               "pca"  → first n_pcs principal components
    n_pcs:     number of PCs if mode="pca"

    Returns:   regressors matrix of shape (T, k)
    """
    # Extract ventral visual parcel time series
    missing = [c for c in ventral_visual_names if c not in ts.columns]
    if missing:
        raise ValueError(f"These ventral visual parcel names are missing from timeseries: {missing[:5]} ...")

    X_vis = ts[ventral_visual_names].values  # T x P_vis

    # Optionally z-score each parcel (not strictly required, but often helpful)
    X_vis = (X_vis - X_vis.mean(axis=0, keepdims=True)) / (X_vis.std(axis=0, ddof=1, keepdims=True) + 1e-8)

    if mode == "mean":
        # Single average regressor
        reg = X_vis.mean(axis=1, keepdims=True)  # T x 1
        return reg
    elif mode == "pca":
        pca = PCA(n_components=min(n_pcs, X_vis.shape[1]))
        reg = pca.fit_transform(X_vis)          # T x k
        explained_var = np.sum(pca.explained_variance_ratio_)
        return reg, explained_var
    else:
        raise ValueError("mode must be 'mean' or 'pca'")

def regress_from_cerebellum(
    ts: pd.DataFrame,
    regressors: np.ndarray,
    cereb_cols: List[str],
    dorsal_only: bool = False,
    dorsal_cereb_labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
    """
    Regress visual-floor regressors from cerebellar parcels.

    ts:               T x 464 DataFrame
    regressors:       T x k matrix (visual-floor regressors)
    cereb_cols:       list of cerebellar column names (e.g., ['region1', ..., 'region32'])
    dorsal_only:      if True, only regress from dorsal cerebellar parcels
    dorsal_cereb_labels:
                      if dorsal_only=True, list of cerebellar parcel names to treat as dorsal.
                      If None and dorsal_only=True, raises an error.

    Returns:          new DataFrame with cerebellar columns replaced by residuals.
    """
    ts_clean = ts.copy()

    # Select which cerebellar columns to clean
    if dorsal_only:
        if dorsal_cereb_labels is None:
            raise ValueError("dorsal_only=True but no dorsal_cereb_labels provided.")
        target_cols = [c for c in cereb_cols if c in dorsal_cereb_labels]
    else:
        target_cols = cereb_cols

    if len(target_cols) == 0:
        raise ValueError("No cerebellar columns selected for cleaning.")

    Y_all = ts[cereb_cols].values      # T x 32
    Y = ts[target_cols].values        # T x N_target

    # Prepare design matrix X = [1, regressors]
    R = regressors
    # Optional: z-score regressors column-wise
    R = (R - R.mean(axis=0, keepdims=True)) / (R.std(axis=0, ddof=1, keepdims=True) + 1e-8)
    X = np.column_stack([np.ones(R.shape[0]), R])  # T x (k+1)

    # Solve least-squares for each cerebellar target (vectorized)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)   # (k+1) x N_target
    Y_hat = X @ beta                               # T x N_target
    Y_resid = Y - Y_hat                            # T x N_target

    # Put residuals back into full cerebellar block
    Y_all_clean = Y_all.copy()
    idx_map = {c: i for i, c in enumerate(cereb_cols)}
    for j, col in enumerate(target_cols):
        Y_all_clean[:, idx_map[col]] = Y_resid[:, j]

    # Replace cerebellar columns in DataFrame
    ts_clean[cereb_cols] = Y_all_clean

    return ts_clean

def clean_cerebellum_visual_bleed(
    ts_path: str,
    schaefer_centroid_path: str,
    mode: Literal["mean", "pca"] = "mean",
    n_pcs: int = 3,
    z_thresh: float = 0.0,
    y_thresh: Optional[float] = None,
    dorsal_only: bool = False,
    dorsal_cereb_labels: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    End-to-end pipeline for visual-floor regression from cerebellum.

    ts_path:                path to subject time series (T x 464, TSV)
    schaefer_centroid_path: path to Schaefer centroid table (400 rows)
    mode:                   'mean' or 'pca' for visual-floor regressors
    n_pcs:                  number of PCs if mode='pca'
    z_thresh:               z threshold for "ventral" (default z < 0)
    y_thresh:               optional y threshold (posterior) if you want
    dorsal_only:            regress only from dorsal cerebellar parcels?
    dorsal_cereb_labels:    list of cerebellar labels to treat as dorsal

    Returns:
      ts_clean: DataFrame with same columns but cleaned cerebellum
      info:     dict with parcel lists used (ventral visual, cerebellar)
    """
    from pathlib import Path
    if isinstance(ts_path, pd.DataFrame):
    # Already a DataFrame: use as-is
        ts = ts_path

    elif isinstance(ts_path, (str, Path)):
        # Assume it's a path to a TSV file
        ts = pd.read_csv(ts_path, sep="\t")
    else:
        raise TypeError(
            f"ts_path must be a pandas DataFrame or a path (str/Path), "
            f"got {type(ts_path).__name__!r} instead."
        )
    # define column groups
    cortical_cols = list(ts.columns[:400])
    subcort_cols  = list(ts.columns[400:432])
    cereb_cols    = list(ts.columns[432:464])  # region1..region32

    # --- load centroids and pick ventral visual ---
    centroids = load_schaefer_centroids(schaefer_centroid_path)
    ventral_vis = get_ventral_visual_parcels(
        centroids,
        z_thresh=z_thresh,
        y_thresh=y_thresh,
    )

    # For safety
    ventral_vis = [v for v in ventral_vis if v in cortical_cols]

    if len(ventral_vis) == 0:
        raise ValueError("No ventral visual parcels matched your timeseries columns; check centroids or naming.")

    # --- build visual-floor regressors ---
    if mode == 'mean':
        regs = build_visual_floor_regressors(
            ts=ts,
            ventral_visual_names=ventral_vis,
            mode=mode,
            n_pcs=n_pcs,
        )
    elif mode == 'pca':
        regs, explained_var = build_visual_floor_regressors(
            ts=ts,
            ventral_visual_names=ventral_vis,
            mode=mode,
            n_pcs=n_pcs,
        )
    # --- regress from cerebellum ---
    ts_clean = regress_from_cerebellum(
        ts=ts,
        regressors=regs,
        cereb_cols=cereb_cols,
        dorsal_only=dorsal_only,
        dorsal_cereb_labels=dorsal_cereb_labels,
    )

    info = {
        "cortical_cols": cortical_cols,
        "subcort_cols": subcort_cols,
        "cereb_cols": cereb_cols,
        "ventral_visual_parcels": ventral_vis,
        "dorsal_cereb_labels_used": dorsal_cereb_labels if dorsal_only else cereb_cols,
        "mode": mode,
        "n_pcs": n_pcs if mode == "pca" else np.nan,
        "explained_var": explained_var if mode == 'pca' else np.nan
    }
    return ts_clean, info

def plot_visual_parcels(
    ventral_visual_parcels: Sequence[str],
    roi_info: "pd.DataFrame",
    atlas_config: Optional["Config"] = None,
    views: Iterable[str] = ("lateral", "posterior", "ventral", "medial"),
    zoom: float = 1.2,
    size: tuple = (3400, 600),
):
    """
    Plot ventral visual parcels on the cortical surface.

    Parameters
    ----------
    ventral_visual_parcels : sequence of str
        List of ROI names (matching `roi_info['roi']`) that are considered ventral visual.
    roi_info : pd.DataFrame
        ROI metadata. Expected columns:
          - 'roi'    : ROI name (matching time-series/atlas labels)
          - 'roi_ix' : integer index into atlas parcels
          - 'network': network label (e.g. 'Vis', 'SomMot', ...)
        The first 400 rows are assumed to be cortical Schaefer parcels.
    atlas_config : Config, optional
        Config object providing `atlas` (used by `plotting.weights_to_vertices`).
        If None, a new `Config()` is instantiated.
    views : iterable of str
        Surface views to show in the plot.
    zoom : float
        Zoom factor for the surfplot Plot.
    size : tuple
        Figure size passed to surfplot (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
    """
    # Use provided config or create a default one
    config = atlas_config if atlas_config is not None else Config()

    # Work on a copy of the cortical part of roi_info (first 400 = Schaefer)
    df = roi_info.iloc[:400, :].copy()

    # Mark ventral visual parcels
    df["ventral"] = 0.0
    df.loc[df["roi"].isin(ventral_visual_parcels), "ventral"] = 1.0

    # Mark non-ventral visual parcels within the visual network
    df["non_ventral"] = 1.0 - df["ventral"]
    df.loc[df["network"] != "Vis", "non_ventral"] = 0.0

    # Map ROI weights to vertices
    ventral_weights = plotting.weights_to_vertices(
        df["ventral"].values,
        config.atlas,
        df["roi_ix"].values,
    )
    non_ventral_weights = plotting.weights_to_vertices(
        df["non_ventral"].values,
        config.atlas,
        df["roi_ix"].values,
    )

    # Colormaps: ventral = red, non-ventral = violet, background = black
    cmap_ventral = LinearSegmentedColormap.from_list("ventral_vis", ["red", "k"], N=2)
    cmap_non_ventral = LinearSegmentedColormap.from_list("non_ventral_vis", ["violet", "k"], N=2)

    # Surfaces and sulcal map
    surfs = get_surfaces()
    sulc = plotting.get_sulc()

    # Build the plot
    p = Plot(
        surfs["lh"],
        surfs["rh"],
        views=list(views),
        layout="row",
        zoom=zoom,
        size=size,
    )

    # Base sulcal map
    p.add_layer(data=sulc, cmap="gray", cbar=False)

    # Ventral parcels (fill + outline)
    p.add_layer(np.nan_to_num(ventral_weights), cmap=cmap_ventral, cbar=False)
    p.add_layer(np.nan_to_num(ventral_weights), as_outline=True, cmap="binary", cbar=False)

    # Non-ventral visual parcels (fill + outline)
    p.add_layer(np.nan_to_num(non_ventral_weights), cmap=cmap_non_ventral, cbar=False)
    p.add_layer(np.nan_to_num(non_ventral_weights), as_outline=True, cmap="binary", cbar=False)

    fig = p.build(colorbar=False)
    return fig
