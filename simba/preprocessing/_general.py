"""General preprocessing functions."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn import preprocessing
from sklearn.utils import sparsefuncs

from ._utils import cal_tf_idf


def _ensure_sparse_matrix(adata):
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)


def log_transform(adata):
    """Apply log1p transform to `adata.X` in-place."""
    _ensure_sparse_matrix(adata)
    adata.X = np.log1p(adata.X)
    return None


def binarize(adata, threshold=1e-5):
    """Binarize `adata.X` in-place with a fixed threshold."""
    _ensure_sparse_matrix(adata)
    adata.X = preprocessing.binarize(adata.X, threshold=threshold, copy=True)


def normalize(adata, method="lib_size", scale_factor=1e4, save_raw=True):
    """Normalize count matrix using library size or TF-IDF."""
    if method not in ["lib_size", "tf_idf"]:
        raise ValueError(f"unrecognized method '{method}'")

    _ensure_sparse_matrix(adata)

    if save_raw:
        adata.layers["raw"] = adata.X.copy()

    if method == "lib_size":
        sparsefuncs.inplace_row_scale(adata.X, 1 / adata.X.sum(axis=1).A)
        adata.X = adata.X * scale_factor
    else:
        adata.X = cal_tf_idf(adata.X)
