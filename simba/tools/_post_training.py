"""Post-training analysis helpers for the minimal scATAC workflow.

中文说明：
该模块负责训练后的核心分析能力：
1) softmax 投影（把 query 实体映射到 reference 细胞空间）
2) 多实体联合嵌入（embed）
3) 实体对比指标（compare_entities）
4) 邻域查询（query）
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.neighbors import KDTree

from ._utils import _gini


def softmax(adata_ref, adata_query, T=0.5, n_top=None, percentile=0):
    """Project query embeddings into reference space with softmax weights.

    中文说明：
    先计算 ref-query 相似度，再按温度系数 T 进行 softmax。
    通过 n_top 或 percentile 稀疏化权重后，将 query 投影到 ref 空间，
    结果写入 `adata_query.layers['softmax']`。
    """
    scores = np.matmul(adata_ref.X, adata_query.X.T)
    scores = scores - scores.max()
    prob = np.exp(scores / T)
    prob = prob / prob.sum(axis=0, keepdims=True)

    if n_top is None:
        thresh = np.percentile(prob, q=percentile, axis=0)
    else:
        thresh = np.sort(prob, axis=0)[::-1, :][n_top - 1, :]

    prob[prob < thresh[None, :]] = 0
    prob = prob / prob.sum(axis=0, keepdims=True)
    adata_query.layers["softmax"] = np.dot(prob.T, adata_ref.X)


def embed(
    adata_ref,
    list_adata_query,
    T=0.5,
    list_T=None,
    percentile=0,
    n_top=None,
    list_percentile=None,
    use_precomputed=False,
):
    """Embed reference and multiple query entities into a shared space.

    中文说明：
    以 adata_ref 作为锚点空间，依次将每个 query 实体做 softmax 投影，
    再按行拼接形成统一 AnnData，obs 中记录 id_dataset=ref/query_i。
    """
    if not isinstance(list_adata_query, list):
        raise TypeError("`list_adata_query` must be list")

    X_all = adata_ref.X.copy()
    obs_all = adata_ref.obs.copy()
    obs_all["id_dataset"] = ["ref"] * adata_ref.n_obs

    for i, adata_query in enumerate(list_adata_query):
        param_T = list_T[i] if list_T is not None else T
        param_percentile = list_percentile[i] if list_percentile is not None else percentile

        if use_precomputed and "softmax" in adata_query.layers:
            print(f"Reading in precomputed softmax-transformed matrix for query data {i};")
        else:
            print(f"Performing softmax transformation for query data {i};")
            softmax(
                adata_ref,
                adata_query,
                T=param_T,
                percentile=param_percentile,
                n_top=n_top,
            )

        X_all = np.vstack((X_all, adata_query.layers["softmax"]))
        obs_query = adata_query.obs.copy()
        obs_query["id_dataset"] = [f"query_{i}"] * adata_query.n_obs
        obs_all = pd.concat([obs_all, obs_query], axis=0)

    return ad.AnnData(X=X_all, obs=obs_all)


def compare_entities(adata_ref, adata_query, n_top_cells=50, T=1):
    """Compute similarity metrics between reference and query entities.

    中文说明：
    生成 adata_cmp 后，在 var 维度给出 max/std/gini/entropy 指标，
    用于筛选细胞类型特异性较强的 motif/kmer/feature 实体。
    """
    X_cmp = np.matmul(adata_ref.X, adata_query.X.T)
    adata_cmp = ad.AnnData(X=X_cmp, obs=adata_ref.obs, var=adata_query.obs)

    adata_cmp.layers["norm"] = X_cmp - np.log(np.exp(X_cmp).mean(axis=0)).reshape(1, -1)
    adata_cmp.layers["softmax"] = np.exp(X_cmp / T) / np.exp(X_cmp / T).sum(axis=0).reshape(1, -1)

    adata_cmp.var["max"] = (
        np.clip(np.sort(adata_cmp.layers["norm"], axis=0)[-n_top_cells:, :], a_min=0, a_max=None).mean(axis=0)
    )
    adata_cmp.var["std"] = np.std(X_cmp, axis=0, ddof=1)
    adata_cmp.var["gini"] = np.array([_gini(adata_cmp.layers["softmax"][:, i]) for i in range(X_cmp.shape[1])])
    adata_cmp.var["entropy"] = entropy(adata_cmp.layers["softmax"])
    return adata_cmp


def query(
    adata,
    obsm="X_umap",
    layer=None,
    metric="euclidean",
    anno_filter=None,
    filters=None,
    entity=None,
    pin=None,
    k=20,
    use_radius=False,
    r=None,
    **kwargs,
):
    """Query nearest entities in embedding space.

    中文说明：
    支持两种查询模式：
    1) kNN（use_radius=False）：返回每个查询点最近的 k 个邻居；
    2) 半径查询（use_radius=True）：返回半径 r 内全部邻居。
    可在 obsm/layer/X 三种空间中执行，并支持按注释字段过滤实体。
    """
    if entity is None and pin is None:
        raise ValueError("One of `entity` and `pin` must be specified")
    if entity is not None and pin is not None:
        print("`entity` will be ignored.")
    if entity is not None:
        entity = np.array(entity).flatten()

    if layer is not None and obsm is not None:
        raise ValueError("Only one of `layer` and `obsm` can be used")

    # 选择查询空间：优先 obsm，其次 layer，最后 adata.X。
    if obsm is not None:
        X = adata.obsm[obsm].copy()
        if pin is None:
            pin = adata[entity, :].obsm[obsm].copy()
    elif layer is not None:
        X = adata.layers[layer].copy()
        if pin is None:
            pin = adata[entity, :].layers[layer].copy()
    else:
        X = adata.X.copy()
        if pin is None:
            pin = adata[entity, :].X.copy()

    pin = np.reshape(np.array(pin), [-1, X.shape[1]])

    # 可选过滤：先按实体注释字段筛选候选集合，再做邻域检索。
    if anno_filter is not None:
        if anno_filter not in adata.obs:
            raise ValueError(f"could not find {anno_filter}")
        if filters is None:
            filters = adata.obs[anno_filter].unique().tolist()
        ids_filters = np.where(np.isin(adata.obs[anno_filter], filters))[0]
    else:
        ids_filters = np.arange(X.shape[0])

    if use_radius:
        kdt = KDTree(X[ids_filters, :], metric=metric, **kwargs)
        if r is None:
            r = np.mean(X.max(axis=0) - X.min(axis=0)) / 5
        ind, dist = kdt.query_radius(pin, r=r, sort_results=True, return_distance=True)

        outputs = []
        for i in range(pin.shape[0]):
            block = adata.obs.iloc[ids_filters].iloc[ind[i], :].copy()
            block["distance"] = dist[i]
            block["query"] = entity[i] if entity is not None else i
            outputs.append(block)
        df_output = pd.concat(outputs, axis=0).sort_values(by="distance") if outputs else pd.DataFrame()
    else:
        kdt = KDTree(X[ids_filters, :], metric=metric, **kwargs)
        dist, ind = kdt.query(pin, k=k, sort_results=True, return_distance=True)

        outputs = []
        for i in range(pin.shape[0]):
            block = adata.obs.iloc[ids_filters].iloc[ind[i, :], :].copy()
            block["distance"] = dist[i, :]
            block["query"] = entity[i] if entity is not None else i
            outputs.append(block)
        df_output = pd.concat(outputs, axis=0).sort_values(by="distance") if outputs else pd.DataFrame()

    # 把查询参数与结果落盘到 adata.uns，保证可复现和可追踪。
    adata.uns["query"] = {
        "params": {
            "obsm": obsm,
            "layer": layer,
            "entity": entity,
            "pin": pin,
            "k": k,
            "use_radius": use_radius,
            "r": r,
        },
        "output": df_output.copy(),
    }
    return df_output
