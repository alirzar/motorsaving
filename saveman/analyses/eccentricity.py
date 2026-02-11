"""Compute subject-level manifold eccentricity across task epochs."""

from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from saveman.config import Config
from saveman.utils import (
    get_files,
    load_gradients,
    schaefer400tian_nettekoven_roi_ix,
)

FIG_DIR = os.path.join(Config().figures, "eccentricity")
os.makedirs(FIG_DIR, exist_ok=True)


_FILENAME_RE = re.compile(
    r"(?P<sub>sub-[^_]+)_(?P<ses>ses-[^_]+)_[^_]*_(?P<epoch>[^_]+)_gradient\.tsv$"
)


def _parse_ids_from_filename(path: str) -> tuple[str, str, str]:
    """
    Parse subject/session/epoch IDs from a gradient filename.

    Falls back to the original underscore-splitting convention if the regex
    doesn't match (to remain backward compatible with any older naming).
    """
    base = os.path.basename(path)
    m = _FILENAME_RE.search(base)
    if m:
        return m.group("sub"), m.group("ses"), m.group("epoch")

    # Backward-compatible fallback: expect at least 4 fields
    parts = base.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unrecognized gradient filename format: {base}")
    sub, ses = parts[0], parts[1]
    epoch = parts[3]
    return sub, ses, epoch


def _read_gradients(fname: str, k: int, sub: str, ses: str, epoch: str) -> pd.DataFrame:
    """Load gradients for one file and attach identifier columns."""
    df = load_gradients(fname, k)

    # Identifier columns
    df["sub"] = sub
    df["ses"] = ses
    df["epoch"] = epoch

    # ROI index mapping (assumed constant ordering across files)
    df["roi_ix"] = schaefer400tian_nettekoven_roi_ix()

    return df


def load_subject_gradients(
    k: int = 3,
    session: Optional[str] = None,
    washout: bool = True,
) -> pd.DataFrame:
    """Load subject gradients into a single DataFrame (long/tidy format).

    Parameters
    ----------
    k : int, optional
        Number of gradients to load, by default 3.
    session : {'ses-01', 'ses-02'}, optional
        If provided, return only one session, by default None.
    washout : bool, optional
        Include washout epochs (if present), by default True.

    Returns
    -------
    pandas.DataFrame
        Subject gradients in long/tidy data format.
    """
    config = Config()
    gradient_dir = config.gradients

    rotation_files = get_files([gradient_dir, "*/*rotation*_gradient.tsv"])
    subject_gradients: List[pd.DataFrame] = []

    for g in rotation_files:
        sub, ses, epoch = _parse_ids_from_filename(g)
        subject_gradients.append(_read_gradients(g, k, sub, ses, epoch))

    if washout:
        washout_files = get_files([gradient_dir, "*/*washout*_gradient.tsv"])
        for g in washout_files:
            sub, ses, epoch = _parse_ids_from_filename(g)
            # Preserve original epoch label while marking it as washout
            epoch = f"washout-{epoch}"
            subject_gradients.append(_read_gradients(g, k, sub, ses, epoch))

    if not subject_gradients:
        raise FileNotFoundError(
            f"No gradient files found under {gradient_dir}. "
            "Check Config().gradients and expected filename patterns."
        )

    subject_gradients_df = pd.concat(subject_gradients, ignore_index=True)

    if session is not None:
        return subject_gradients_df.query("ses == @session").reset_index(drop=True)

    return subject_gradients_df


def compute_eccentricity(data: pd.DataFrame) -> pd.DataFrame:
    """Compute ROI eccentricity as distance from the epoch-specific centroid.

    Parameters
    ----------
    data : pandas.DataFrame
        Long/tidy subject gradient data. Must include columns:
        ['sub', 'ses', 'epoch'] and gradient columns named like 'g1', 'g2', ...

    Returns
    -------
    pandas.DataFrame
        Same data with an added 'distance' column.
    """
    grad_cols = [c for c in data.columns if c.startswith("g")]
    if not grad_cols:
        raise ValueError("No gradient columns found (expected columns starting with 'g').")

    # Compute per-(sub,ses,epoch) centroid for each gradient column and subtract
    centroids = data.groupby(["sub", "ses", "epoch"], sort=False)[grad_cols].transform("mean")
    diffs = data[grad_cols].to_numpy() - centroids.to_numpy()
    distances = np.linalg.norm(diffs, axis=1)

    out = data.copy()
    out["distance"] = distances
    return out

def main() -> None:
    """Entry point: load gradients, compute eccentricity, and write results TSV."""
    config = Config()

    gradients = load_subject_gradients(k=config.k)
    gradients = compute_eccentricity(gradients)

    out_path = os.path.join(config.results, "subject_gradients.tsv")
    gradients.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
