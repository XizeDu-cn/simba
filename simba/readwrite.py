"""Read/write utilities for the minimal SIMBA workflow.

中文说明：
负责训练结果读取（embedding/config/stats）与 BED 导出。
重点是 alias 与原始实体 ID 的双向映射，保证结果可解释。
"""

import json
import os
from pathlib import Path

import pandas as pd
from anndata import read_h5ad, read_hdf

from ._settings import settings


def read_embedding(
    path_emb=None,
    path_entity=None,
    convert_alias=True,
    path_entity_alias=None,
    prefix=None,
    num_epochs=None,
):
    """Read entity embeddings produced by PBG training.

    中文说明：
    PBG 的 embedding 文件使用 alias（如 C.0）存储实体顺序。
    当 convert_alias=True 时，通过 entity_alias.txt 反查回原始实体 ID。
    """
    pbg_params = settings.pbg_params
    path_emb = path_emb or pbg_params["checkpoint_path"]
    path_entity = path_entity or pbg_params["entity_path"]
    num_epochs = num_epochs or pbg_params["num_epochs"]
    prefix = prefix or []

    if not isinstance(prefix, list):
        raise TypeError("`prefix` must be list")

    alias_table = None
    if convert_alias:
        # alias_table 结构：index=alias，列 id=原始实体名。
        alias_root = path_entity_alias or Path(path_emb).parent.as_posix()
        alias_table = pd.read_csv(
            os.path.join(alias_root, "entity_alias.txt"),
            header=0,
            index_col=0,
            sep="\t",
        )
        alias_table["id"] = alias_table.index
        alias_table.index = alias_table["alias"].values

    dict_adata = {}
    for filename in os.listdir(path_emb):
        if not filename.startswith("embeddings"):
            continue

        entity_type = filename.split("_")[1]
        if prefix and entity_type not in prefix:
            continue

        adata = read_hdf(
            os.path.join(path_emb, f"embeddings_{entity_type}_0.v{num_epochs}.h5"),
            key="embeddings",
        )
        with open(os.path.join(path_entity, f"entity_names_{entity_type}_0.json"), "rt") as fp:
            names_entity = json.load(fp)

        if convert_alias:
            names_entity = alias_table.loc[names_entity, "id"].tolist()

        adata.obs.index = names_entity
        dict_adata[entity_type] = adata

    return dict_adata


def load_pbg_config(path=None):
    """Load PBG config into global settings.

    中文说明：
    读取训练后 model/config.json，覆盖 settings.pbg_params，
    便于后续 read_embedding/query 使用一致上下文。
    """
    path = path or settings.pbg_params["checkpoint_path"]
    path = os.path.normpath(path)
    with open(os.path.join(path, "config.json"), "rt") as fp:
        pbg_params = json.load(fp)
    settings.set_pbg_params(config=pbg_params)


def load_graph_stats(path=None):
    """Load graph statistics into global settings."""
    if path is None:
        path = Path(settings.pbg_params["entity_path"]).parent.parent.as_posix()
    path = os.path.normpath(path)
    with open(os.path.join(path, "graph_stats.json"), "rt") as fp:
        graph_stats = json.load(fp)
    settings.graph_stats[os.path.basename(path)] = graph_stats


def write_bed(adata, use_top_pcs=True, filename=None):
    """Write peaks as BED records (`chr`, `start`, `end`).

    中文说明：
    默认只输出 top_pcs 对应 peak，和教程保持一致；
    若 use_top_pcs=False，则导出全部 peak。
    """
    filename = filename or os.path.join(settings.workdir, "peaks.bed")

    for key in ["chr", "start", "end"]:
        if key not in adata.var:
            raise ValueError(f"could not find {key} in `adata.var_keys()`")

    if use_top_pcs:
        if "top_pcs" not in adata.var:
            raise ValueError("please run `si.pp.select_pcs_features()` first")
        peaks = adata.var.loc[adata.var["top_pcs"], ["chr", "start", "end"]]
    else:
        peaks = adata.var[["chr", "start", "end"]]

    peaks.to_csv(filename, sep="\t", header=False, index=False)
    out_dir, out_file = os.path.split(filename)
    print(f'"{out_file}" was written to "{out_dir}".')
