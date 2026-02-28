"""Quality-control helpers for the minimal scATAC workflow."""

import numpy as np
from scipy.sparse import csr_matrix, issparse


def _ensure_sparse_matrix(adata):
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)


def _obs_qc_arrays(adata, expr_cutoff):
    n_counts = adata.obs["n_counts"] if "n_counts" in adata.obs else adata.X.sum(axis=1).A1
    n_peaks = adata.obs["n_peaks"] if "n_peaks" in adata.obs else (adata.X >= expr_cutoff).sum(axis=1).A1
    pct_peaks = adata.obs["pct_peaks"] if "pct_peaks" in adata.obs else n_peaks / adata.shape[1]
    return n_counts, n_peaks, pct_peaks


def _var_qc_arrays(adata, expr_cutoff):
    n_counts = adata.var["n_counts"] if "n_counts" in adata.var else adata.X.sum(axis=0).A1
    n_cells = adata.var["n_cells"] if "n_cells" in adata.var else (adata.X >= expr_cutoff).sum(axis=0).A1
    pct_cells = adata.var["pct_cells"] if "pct_cells" in adata.var else n_cells / adata.shape[0]
    return n_counts, n_cells, pct_cells


def cal_qc_atac(adata, expr_cutoff=1):
    """Compute ATAC-specific QC metrics for cells and peaks."""
    _ensure_sparse_matrix(adata)

    n_counts_var = adata.X.sum(axis=0).A1
    n_cells = (adata.X >= expr_cutoff).sum(axis=0).A1
    adata.var["n_counts"] = n_counts_var
    adata.var["n_cells"] = n_cells
    adata.var["pct_cells"] = n_cells / adata.shape[0]

    n_counts_obs = adata.X.sum(axis=1).A1
    n_peaks = (adata.X >= expr_cutoff).sum(axis=1).A1
    adata.obs["n_counts"] = n_counts_obs
    adata.obs["n_peaks"] = n_peaks
    adata.obs["pct_peaks"] = n_peaks / adata.shape[1]


def filter_cells_atac(
    adata,
    min_n_peaks=None,
    max_n_peaks=None,
    min_pct_peaks=None,
    max_pct_peaks=None,
    min_n_counts=None,
    max_n_counts=None,
    expr_cutoff=1,
):
    """Filter cells in scATAC data based on QC thresholds."""
    _ensure_sparse_matrix(adata)

    n_counts, n_peaks, pct_peaks = _obs_qc_arrays(adata, expr_cutoff)
    adata.obs["n_counts"] = n_counts
    adata.obs["n_peaks"] = n_peaks
    adata.obs["pct_peaks"] = pct_peaks

    print("Before filtering: ")
    print(f"{adata.shape[0]} cells, {adata.shape[1]} peaks")

    filters = [min_n_peaks, max_n_peaks, min_pct_peaks, max_pct_peaks, min_n_counts, max_n_counts]
    if all(x is None for x in filters):
        print("No filtering")
        return None

    subset = np.ones(adata.n_obs, dtype=bool)
    if min_n_peaks is not None:
        print("Filter cells based on min_n_peaks")
        subset &= n_peaks >= min_n_peaks
    if max_n_peaks is not None:
        print("Filter cells based on max_n_peaks")
        subset &= n_peaks <= max_n_peaks
    if min_pct_peaks is not None:
        print("Filter cells based on min_pct_peaks")
        subset &= pct_peaks >= min_pct_peaks
    if max_pct_peaks is not None:
        print("Filter cells based on max_pct_peaks")
        subset &= pct_peaks <= max_pct_peaks
    if min_n_counts is not None:
        print("Filter cells based on min_n_counts")
        subset &= n_counts >= min_n_counts
    if max_n_counts is not None:
        print("Filter cells based on max_n_counts")
        subset &= n_counts <= max_n_counts

    adata._inplace_subset_obs(subset)
    print("After filtering out low-quality cells: ")
    print(f"{adata.shape[0]} cells, {adata.shape[1]} peaks")
    return None


def filter_peaks(
    adata,
    min_n_cells=5,
    max_n_cells=None,
    min_pct_cells=None,
    max_pct_cells=None,
    min_n_counts=None,
    max_n_counts=None,
    expr_cutoff=1,
):
    """Filter peaks based on feature-level QC thresholds."""
    _ensure_sparse_matrix(adata)

    n_counts, n_cells, pct_cells = _var_qc_arrays(adata, expr_cutoff)
    adata.var["n_counts"] = n_counts
    adata.var["n_cells"] = n_cells
    adata.var["pct_cells"] = pct_cells

    print("Before filtering: ")
    print(f"{adata.shape[0]} cells, {adata.shape[1]} peaks")

    filters = [min_n_cells, max_n_cells, min_pct_cells, max_pct_cells, min_n_counts, max_n_counts]
    if all(x is None for x in filters):
        print("No filtering")
        return None

    subset = np.ones(adata.n_vars, dtype=bool)
    if min_n_cells is not None:
        print("Filter peaks based on min_n_cells")
        subset &= n_cells >= min_n_cells
    if max_n_cells is not None:
        print("Filter peaks based on max_n_cells")
        subset &= n_cells <= max_n_cells
    if min_pct_cells is not None:
        print("Filter peaks based on min_pct_cells")
        subset &= pct_cells >= min_pct_cells
    if max_pct_cells is not None:
        print("Filter peaks based on max_pct_cells")
        subset &= pct_cells <= max_pct_cells
    if min_n_counts is not None:
        print("Filter peaks based on min_n_counts")
        subset &= n_counts >= min_n_counts
    if max_n_counts is not None:
        print("Filter peaks based on max_n_counts")
        subset &= n_counts <= max_n_counts

    adata._inplace_subset_var(subset)
    print("After filtering out low-expressed peaks: ")
    print(f"{adata.shape[0]} cells, {adata.shape[1]} peaks")
    return None
