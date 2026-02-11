"""Functional PCA (FPCA) on motor adaptation learning curves."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr
import skfda

from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.representation.basis import BSplineBasis

from saveman.config import Config
from saveman.analyses.plotting import set_plotting


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _savefig(fig: plt.Figure, path_no_ext: str, dpi: int = 300) -> None:
    out = path_no_ext if path_no_ext.lower().endswith(".png") else (path_no_ext + ".png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fill_data(
    df: pd.DataFrame,
    angle_col: str = "error",
    subject_col: str = "sub",
    trial_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a subjects × trials matrix with gap-filled values.

    If `trial_col` is provided and exists, curves are aligned on that explicit trial axis.
    Missing values are interpolated along the trial axis, then padded at the edges.
    """
    df = df.copy()

    if trial_col and trial_col in df.columns:
        wide = df[[subject_col, trial_col, angle_col]].pivot_table(
            index=subject_col, columns=trial_col, values=angle_col, aggfunc="mean"
        )
    else:
        # Fallback: within-subject row order (less safe than an explicit trial axis).
        df["__trial_idx__"] = df.groupby(subject_col).cumcount()
        wide = df[[subject_col, "__trial_idx__", angle_col]].pivot_table(
            index=subject_col, columns="__trial_idx__", values=angle_col, aggfunc="mean"
        )

    # Sort trial axis numerically when possible
    try:
        wide.columns = pd.to_numeric(wide.columns)
    except Exception:
        pass
    wide = wide.sort_index(axis=1).astype(float)

    # Interpolate along trials then pad edges
    wide = wide.interpolate(axis=1, limit_direction="both").bfill(axis=1).ffill(axis=1)
    return wide


# ----------------------------
# FPCA core
# ----------------------------

def epoch_fpca(
    df: pd.DataFrame,
    session: Sequence[str],
    fig_dir: str,
    res_dir: str,
    prefix: str,
    n_basis: int = 17,
    n_components: int = 1,
    trial: str = "trial",
    fmin: float = 0.0,
    fmax: float = 3.0,
) -> FPCA:
    """Run FPCA for the provided sessions and save scores + FPC1 deformation plot."""
    _ensure_dir(fig_dir)
    _ensure_dir(res_dir)

    epoch_behav = df.query('epoch not in ["base"] & ses in @session').copy()
    if epoch_behav.empty:
        raise ValueError("No rows found after filtering for requested sessions/epochs.")

    # Concatenate sessions along trial axis (avoid averaging ses-01 + ses-02).
    if trial in epoch_behav.columns and isinstance(session, (list, tuple)) and len(session) > 1:
        ses_order = {ses: i for i, ses in enumerate(session)}
        epoch_behav["_ses_order"] = epoch_behav["ses"].map(ses_order).astype(int)

        epoch_behav = epoch_behav.sort_values(["sub", "_ses_order", trial])
        epoch_behav["_ses_trial_key"] = epoch_behav["ses"].astype(str) + "||" + epoch_behav[trial].astype(str)
        epoch_behav["_trial_concat"] = (
            epoch_behav.groupby("sub")["_ses_trial_key"].transform(lambda s: pd.factorize(s, sort=False)[0]).astype(int)
        )
        trial_for_fpca = "_trial_concat"
    else:
        trial_for_fpca = trial

    data = fill_data(epoch_behav, angle_col="error", subject_col="sub", trial_col=trial_for_fpca)
    grid_points = data.columns.values.astype(float)
    data_matrix = data.values

    fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
    basis = BSplineBasis(n_basis=n_basis, order=4)
    basis_fd = fd.to_basis(basis)

    fpca = FPCA(
        n_components=n_components,
        components_basis=BSplineBasis(n_basis=n_basis, order=4),
        regularization=L2Regularization(LinearDifferentialOperator(2)),
    )
    scores = fpca.fit_transform(basis_fd)

    # Flip only FPC1 scores (sign convention)
    if n_components >= 1:
        scores[:, 0] *= -1

    scores_df = pd.DataFrame(scores, index=data.index, columns=[f"FPC{c+1}" for c in range(n_components)])
    scores_df = scores_df.reset_index().rename(columns={"index": "sub"})
    scores_df.to_csv(os.path.join(res_dir, f"{prefix}.tsv"), sep="\t", index=False)

    # Deformation plot (FPC1): mean ± factor * sqrt(eigenvalue) * eigenfunction
    factors = np.arange(fmin, fmax, 0.01).tolist()
    mean_fd = basis_fd.mean()
    mean_grid = mean_fd.to_grid(grid_points).data_matrix.squeeze()

    comp_fd = fpca.components_[0]
    comp_grid = comp_fd.to_grid(grid_points).data_matrix.squeeze()
    scale = float(np.sqrt(fpca.explained_variance_[0]))

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap_r = cmr.get_sub_cmap("RdYlGn", 0, 0.5)
    cmap_g = cmr.get_sub_cmap("RdYlGn", 0.5, 1)

    for i, factor in enumerate(factors):
        red = cmap_r(1 - (i + 1) / len(factors))
        green = cmap_g((i + 1) / len(factors))
        ax.plot(grid_points, mean_grid + factor * scale * comp_grid, color=green, linewidth=1)
        ax.plot(grid_points, mean_grid - factor * scale * comp_grid, color=red, linewidth=1)

    ax.plot(grid_points, mean_grid, color="black", linewidth=2)
    ax.set_xlabel("Trial", fontsize=12, fontweight="bold")
    ax.set_ylabel("Angular Error", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _savefig(fig, os.path.join(fig_dir, f"{prefix}_fpca_fpc1"))
    return fpca


def plot_loadingxlearning(data: pd.DataFrame, scores: pd.DataFrame, out_dir: str, angle: str = "error") -> None:
    """Plot each subject's learning curve colored by FPCA loading (FPC1)."""
    _ensure_dir(out_dir)

    if "FPC1" not in scores.columns:
        raise ValueError("`scores` must contain a 'FPC1' column.")
    if "sub" not in data.columns or "sub" not in scores.columns:
        raise ValueError("Both `data` and `scores` must contain a 'sub' column.")

    min_score = float(scores["FPC1"].min())
    max_score = float(scores["FPC1"].max())
    range_score = max(max_score - min_score, 1e-12)

    subjects = data["sub"].unique()
    n_trial = len(data[data["sub"] == subjects[0]])

    df = data.copy()
    df["Trial"] = np.tile(np.arange(n_trial), len(subjects)) + 1

    cmap = cmr.get_sub_cmap("RdYlGn", 0, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for subject in subjects:
        sub_data = df[df["sub"] == subject]
        sc = scores.loc[scores["sub"] == subject, "FPC1"]
        if sc.empty:
            continue
        color = cmap(float((sc.iloc[0] - min_score) / range_score))
        ax.plot(sub_data["Trial"], sub_data[angle], color=color, linewidth=1)

    ax.set_xlabel("Trial Bin", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, os.path.join(out_dir, "loadingxlearning_color_coded"))

    # “Sample subjects” panel (max/min + optional example)
    max_sub = scores.loc[scores["FPC1"].idxmax(), "sub"]
    min_sub = scores.loc[scores["FPC1"].idxmin(), "sub"]
    sample_subs = [max_sub, min_sub]

    fig, ax = plt.subplots(figsize=(7, 5))
    for subject in subjects:
        sub_data = df[df["sub"] == subject]
        ax.plot(sub_data["Trial"], sub_data[angle], color="lightgrey", linewidth=1)

    for subject, color in [(max_sub, "green"), (min_sub, "red")]:
        sub_data = df[df["sub"] == subject]
        ax.plot(sub_data["Trial"], sub_data[angle], color=color, linewidth=2)

    mean_curve = df.groupby("Trial")[angle].mean()
    ax.plot(mean_curve.index, mean_curve.values, color="k", linewidth=2)

    ax.set_xlabel("Trial Bin", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, os.path.join(out_dir, "loadingxlearning_sample_subjects"))


def nbasis_var_trade_off(
    df: pd.DataFrame,
    epochs,
    blocks,
    fig_dir: str,
    res_dir: str,
    n_components: int = 1,
    angle_col: str = "error",
    trial_col: str = "trial",
) -> None:
    """Explore n_basis vs explained variance trade-off."""
    _ensure_dir(os.path.join(fig_dir, "fpca_test"))
    _ensure_dir(res_dir)

    n_basis_range = np.arange(7, 30)

    for epoch, blockNo in zip(epochs, blocks):
        var = []
        for n_basis in n_basis_range:
            epoch_behav = df.query("blockNo in @blockNo").copy()

            data_df = fill_data(
                epoch_behav,
                angle_col=angle_col,
                subject_col="sub",
                trial_col=trial_col if trial_col in epoch_behav.columns else None,
            )

            grid_points = data_df.columns.values.astype(float)
            data_matrix = data_df.values

            fd = skfda.FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
            basis = BSplineBasis(n_basis=n_basis, order=4)
            basis_fd = fd.to_basis(basis)

            fpca = FPCA(
                n_components=n_components,
                components_basis=BSplineBasis(n_basis=n_basis, order=4),
                regularization=L2Regularization(LinearDifferentialOperator(2)),
            )
            fpca.fit_transform(basis_fd)
            var.append(fpca.explained_variance_ratio_)

        var_df = pd.DataFrame({"n_basis": n_basis_range, "explained_var": var})
        var_df.to_csv(os.path.join(res_dir, f"{epoch}_nbasis-var_trade-off.tsv"), sep="\t", index=False)


def main() -> None:
    config = Config()
    fig_dir = os.path.join(config.figures, "fpca")
    res_dir = os.path.join(config.results, "fpca")
    _ensure_dir(fig_dir)
    _ensure_dir(res_dir)

    # Apply lab plotting style only for script execution
    set_plotting()

    ses = [["ses-01", "ses-02"]]
    name = ["D1D2"]
    n_basis = 17

    df = pd.read_csv(os.path.join(config.resources, "subject_behavior_bin.csv"))

    var_angular = []
    for s, n in zip(ses, name):
        prefix = f"{n}-{n_basis}bases_angular_error_bin"
        res = epoch_fpca(
            df,
            s,
            fig_dir,
            res_dir,
            prefix,
            n_components=2,
            n_basis=n_basis,
            trial="trial_bin",
            fmax=1.2,
        )
        var_angular.append(res.explained_variance_ratio_)

    var_df = pd.DataFrame({"ses": ses, "var_angular_error": var_angular, "n_basis": n_basis})
    var_df.to_csv(os.path.join(res_dir, "explained_var_ratio_bin.tsv"), sep="\t", index=False)

    # Example plot: scores × learning curves
    epoch_behav = df.query("trial_bin > 15").copy()
    epoch_behav["error"] = (180 * epoch_behav["error"]) / np.pi  # radians -> degrees

    scores = pd.read_csv(os.path.join(res_dir, "D1D2-17bases_angular_error_bin.tsv"), sep="\t").iloc[:, :2]
    out_dir = os.path.join(fig_dir, "D1D2_score_plots")
    plot_loadingxlearning(epoch_behav.query("ses in ['ses-01', 'ses-02']"), scores, out_dir)


if __name__ == "__main__":
    main()
