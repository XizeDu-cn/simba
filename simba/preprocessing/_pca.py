"""Principal component analysis helpers.

中文说明：
本模块实现“先降维，再按肘部法选 PC 和关键特征”的标准流程，
并把所有结果写入 `adata.uns['pca']` 与 `adata.var['top_pcs']`。
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD

from ._utils import locate_elbow


def pca(
    adata,
    n_components=50,
    algorithm="randomized",
    n_iter=5,
    random_state=2021,
    tol=0.0,
    feature=None,
    **kwargs,
):
    """Perform PCA via TruncatedSVD and store results in `adata`.

    中文说明：
    这里使用 TruncatedSVD 以兼容稀疏矩阵输入，避免 densify 带来的内存压力。
    """
    if feature is None:
        X = adata.X.copy()
    else:
        X = adata[:, adata.var[feature]].X.copy()

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm=algorithm,
        n_iter=n_iter,
        random_state=random_state,
        tol=tol,
        **kwargs,
    )
    svd.fit(X)

    adata.obsm["X_pca"] = svd.transform(X)
    adata.uns["pca"] = {
        "n_pcs": n_components,
        "PCs": svd.components_.T,
        "variance": svd.explained_variance_,
        "variance_ratio": svd.explained_variance_ratio_,
    }


def select_pcs(
    adata,
    n_pcs=None,
    S=1,
    curve="convex",
    direction="decreasing",
    online=False,
    min_elbow=None,
    **kwargs,
):
    """Select top PCs based on variance ratio elbow.

    中文说明：
    若 n_pcs 未指定，使用 locate_elbow 在解释方差曲线上自动找拐点。
    """
    if n_pcs is None:
        n_components = adata.obsm["X_pca"].shape[1]
        if min_elbow is None:
            min_elbow = n_components / 10
        n_pcs = locate_elbow(
            range(n_components),
            adata.uns["pca"]["variance_ratio"],
            S=S,
            curve=curve,
            min_elbow=min_elbow,
            direction=direction,
            online=online,
            **kwargs,
        )

    adata.uns["pca"]["n_pcs"] = n_pcs


def select_pcs_features(
    adata,
    S=1,
    curve="convex",
    direction="decreasing",
    online=False,
    min_elbow=None,
    **kwargs,
):
    """Select features associated with each retained PC.

    中文说明：
    对每个 PC 的 loading 绝对值降序曲线做肘部检测，
    选出该 PC 贡献最大的特征；最终合并得到 `var['top_pcs']`。
    """
    n_pcs = adata.uns["pca"]["n_pcs"]
    n_features = adata.uns["pca"]["PCs"].shape[0]
    if min_elbow is None:
        min_elbow = n_features / 6

    adata.uns["pca"]["features"] = {}
    selected_feature_ids = []

    for i in range(n_pcs):
        # 对每个 PC 独立寻找特征截断点（elbow），减少手工阈值依赖。
        loadings = np.sort(np.abs(adata.uns["pca"]["PCs"][:, i]))[::-1]
        elbow = locate_elbow(
            range(n_features),
            loadings,
            S=S,
            min_elbow=min_elbow,
            curve=curve,
            direction=direction,
            online=online,
            **kwargs,
        )
        ids_i = list(np.argsort(np.abs(adata.uns["pca"]["PCs"][:, i]))[::-1][:elbow])
        adata.uns["pca"]["features"][f"pc_{i}"] = ids_i
        selected_feature_ids.extend(ids_i)
        print(f"#features selected from PC {i}: {len(ids_i)}")

    adata.var["top_pcs"] = False
    adata.var.loc[adata.var_names[np.unique(selected_feature_ids)], "top_pcs"] = True
    print(f"#features in total: {adata.var['top_pcs'].sum()}")
