"""PyTorch-BigGraph (PBG) graph construction and training.

中文说明：
本文件负责把 AnnData 组织为 PBG 可读的边表/实体别名，并启动训练。
核心约定是三类关系：
1) C -> P (r0, weight=1.0)
2) P -> M (r1, weight=0.2)
3) P -> K (r2, weight=0.02)
"""

import json
import os
from pathlib import Path

import attr
import numpy as np
import pandas as pd
from torchbiggraph.config import ConfigFileLoader, add_to_sys_path
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer, set_logging_verbosity, setup_logging

from .._settings import settings


# Relation templates for the scATAC workflow.
RELATION_BLUEPRINTS = {
    "CP": {"lhs": "C", "rhs": "P", "weight": 1.0},
    "PM": {"lhs": "P", "rhs": "M", "weight": 0.2},
    "PK": {"lhs": "P", "rhs": "K", "weight": 0.02},
}


def _effective_top_pcs(default_flag, override_flag):
    """解析 use_top_pcs 的全局开关与局部覆盖开关。"""
    return default_flag if override_flag is None else override_flag


def _select_matrix(adata, use_top_pcs):
    """按需选择 top_pcs 特征子集，保证与教程逻辑一致。"""
    if use_top_pcs:
        if "top_pcs" not in adata.var:
            raise ValueError("`top_pcs` not found in adata.var; run `si.pp.select_pcs_features()` first.")
        return adata[:, adata.var["top_pcs"]].copy()
    return adata.copy()


def _alias_frame(index, prefix):
    return pd.DataFrame(index=index, data={"alias": [f"{prefix}.{i}" for i in range(len(index))]})


def _init_graph_dirs(dirname):
    """Create graph output directories and reset graph-related PBG settings.

    中文说明：
    每次 gen_graph 都会重置 settings.pbg_params 中与图结构相关的字段，
    避免上一次运行的实体/关系残留影响本次构图。
    """
    path_graph = os.path.join(settings.workdir, "pbg", dirname)
    os.makedirs(path_graph, exist_ok=True)

    settings.pbg_params["entity_path"] = os.path.join(path_graph, "input/entity")
    settings.pbg_params["edge_paths"] = [os.path.join(path_graph, "input/edge")]
    settings.pbg_params["entities"] = {}
    settings.pbg_params["relations"] = []
    return path_graph


def _build_entity_aliases(entity_ids, prefixes):
    """Build alias table for each entity type and update PBG entity config.

    中文说明：
    PBG 训练文件要求节点用紧凑别名（如 C.0 / P.123），
    这里同时生成“原始 ID -> alias”映射，并注册实体分区配置。
    """
    alias_tables = {}
    entity_alias = pd.DataFrame(columns=["alias"])

    for kind, ids in entity_ids.items():
        if len(ids) == 0:
            continue
        prefix = prefixes[kind]
        table = _alias_frame(ids, prefix)
        alias_tables[kind] = table
        settings.pbg_params["entities"][prefix] = {"num_partitions": 1}
        entity_alias = pd.concat([entity_alias, table], axis=0)

    return alias_tables, entity_alias


def _apply_pbg_ids(adata_original, adata_view, obs_alias, var_alias):
    """Write generated PBG aliases back to the original AnnData object.

    中文说明：
    该步骤让上游 AnnData 保留 pbg_id，可用于后续 trace/query 可解释性分析。
    """
    adata_original.obs["pbg_id"] = ""
    adata_original.var["pbg_id"] = ""
    adata_original.obs.loc[adata_view.obs_names, "pbg_id"] = obs_alias.loc[adata_view.obs_names, "alias"].copy()
    adata_original.var.loc[adata_view.var_names, "pbg_id"] = var_alias.loc[adata_view.var_names, "alias"].copy()


def _relation_edges(adata_view, source_alias, target_alias, relation_name):
    row_idx, col_idx = adata_view.X.nonzero()
    return pd.DataFrame(
        {
            "source": source_alias.loc[adata_view.obs_names[row_idx], "alias"].values,
            "relation": relation_name,
            "destination": target_alias.loc[adata_view.var_names[col_idx], "alias"].values,
        }
    )


def _register_relation(name, lhs, rhs, weight):
    """向 settings.pbg_params 注册一条关系配置。"""
    settings.pbg_params["relations"].append(
        {"name": name, "lhs": lhs, "rhs": rhs, "operator": "none", "weight": weight}
    )


def _collect_views_and_ids(list_CP, list_PM, list_PK, cp_top, pm_top, pk_top):
    """Create matrix views and collect union IDs for each entity type.

    中文说明：
    这里采用“统一 union”策略来收集 cells/peaks/motifs/kmers，
    不再引入多数据集 cell 字典分区逻辑，以提升可读性并满足单 h5ad 场景。
    """
    views = {"CP": [], "PM": [], "PK": []}
    entity_ids = {
        "cells": pd.Index([]),
        "peaks": pd.Index([]),
        "motifs": pd.Index([]),
        "kmers": pd.Index([]),
    }

    for adata_original in list_CP:
        adata_view = _select_matrix(adata_original, cp_top)
        # 每条记录保留 (原始对象, 用于构边的视图)
        views["CP"].append((adata_original, adata_view))
        entity_ids["cells"] = entity_ids["cells"].union(adata_view.obs.index)
        entity_ids["peaks"] = entity_ids["peaks"].union(adata_view.var.index)

    for adata_original in list_PM:
        adata_view = _select_matrix(adata_original, pm_top)
        views["PM"].append((adata_original, adata_view))
        entity_ids["peaks"] = entity_ids["peaks"].union(adata_view.obs.index)
        entity_ids["motifs"] = entity_ids["motifs"].union(adata_view.var.index)

    for adata_original in list_PK:
        adata_view = _select_matrix(adata_original, pk_top)
        views["PK"].append((adata_original, adata_view))
        entity_ids["peaks"] = entity_ids["peaks"].union(adata_view.obs.index)
        entity_ids["kmers"] = entity_ids["kmers"].union(adata_view.var.index)

    return views, entity_ids


def _emit_relation_edges(views, alias_tables, prefixes):
    """Generate all relation edges and return edge table + stats table.

    中文说明：
    relation_plan 明确规定了边的方向与语义，输出包含：
    - df_edges: PBG 输入边表（source/relation/destination）
    - relation_table: 人类可读的关系统计表（含权重和边数）
    """
    relation_stats = []
    edges_all = []
    relation_id = 0

    # 注意：顺序决定 relation 名称（r0/r1/r2...），与教程输出保持一致。
    relation_plan = [
        ("CP", "cells", "peaks"),
        ("PM", "peaks", "motifs"),
        ("PK", "peaks", "kmers"),
    ]

    for relation_kind, source_kind, target_kind in relation_plan:
        if relation_kind not in views:
            continue

        blueprint = RELATION_BLUEPRINTS[relation_kind]
        lhs = prefixes[source_kind]
        rhs = prefixes[target_kind]

        for adata_original, adata_view in views[relation_kind]:
            relation_name = f"r{relation_id}"
            df_edges = _relation_edges(
                adata_view,
                alias_tables[source_kind],
                alias_tables[target_kind],
                relation_name,
            )
            n_edges = int(df_edges.shape[0])

            print(f"relation{relation_id}: source: {lhs}, destination: {rhs}\\n#edges: {n_edges}")
            relation_stats.append(
                {
                    "relation": relation_name,
                    "source": lhs,
                    "destination": rhs,
                    "weight": blueprint["weight"],
                    "n_edges": n_edges,
                }
            )
            _register_relation(relation_name, lhs, rhs, blueprint["weight"])
            _apply_pbg_ids(adata_original, adata_view, alias_tables[source_kind], alias_tables[target_kind])

            edges_all.append(df_edges)
            relation_id += 1

    df_edges = (
        pd.concat(edges_all, axis=0, ignore_index=True)
        if edges_all
        else pd.DataFrame(columns=["source", "relation", "destination"])
    )
    return df_edges, pd.DataFrame(relation_stats)


def gen_graph(
    list_CP=None,
    list_PM=None,
    list_PK=None,
    prefix_C="C",
    prefix_P="P",
    prefix_M="M",
    prefix_K="K",
    copy=False,
    dirname="graph0",
    use_top_pcs=True,
    use_top_pcs_CP=None,
    use_top_pcs_PM=None,
    use_top_pcs_PK=None,
):
    """Generate PBG graph from CP/PM/PK matrices.

    The minimal build uses simple union indexing for all entity types.
    中文说明：
    函数目标是稳定产出 3 个关键文件：
    1) pbg_graph.txt   训练边表
    2) entity_alias.txt 实体别名映射
    3) graph_stats.json 关系统计信息
    """

    list_CP = list_CP or []
    list_PM = list_PM or []
    list_PK = list_PK or []
    if len(list_CP) + len(list_PM) + len(list_PK) == 0:
        return "No graph is generated"

    path_graph = _init_graph_dirs(dirname)

    cp_top = _effective_top_pcs(use_top_pcs, use_top_pcs_CP)
    pm_top = _effective_top_pcs(use_top_pcs, use_top_pcs_PM)
    pk_top = _effective_top_pcs(use_top_pcs, use_top_pcs_PK)

    views, entity_ids = _collect_views_and_ids(list_CP, list_PM, list_PK, cp_top, pm_top, pk_top)
    prefixes = {
        "cells": prefix_C,
        "peaks": prefix_P,
        "motifs": prefix_M,
        "kmers": prefix_K,
    }
    alias_tables, entity_alias = _build_entity_aliases(entity_ids, prefixes)

    df_edges, relation_table = _emit_relation_edges(views, alias_tables, prefixes)

    print(f"Total number of edges: {df_edges.shape[0]}")

    stats = {
        row["relation"]: {
            "source": row["source"],
            "destination": row["destination"],
            "n_edges": int(row["n_edges"]),
        }
        for _, row in relation_table.iterrows()
    }
    stats["n_edges"] = int(df_edges.shape[0])
    settings.graph_stats[dirname] = stats

    print(f'Writing graph file "pbg_graph.txt" to "{path_graph}" ...')
    df_edges.to_csv(os.path.join(path_graph, "pbg_graph.txt"), header=False, index=False, sep="\t")
    entity_alias.to_csv(os.path.join(path_graph, "entity_alias.txt"), header=True, index=True, sep="\t")
    # 额外输出一份人类可读关系清单，便于快速审阅图结构。
    relation_table.to_csv(os.path.join(path_graph, "relations.tsv"), header=True, index=False, sep="\t")

    with open(os.path.join(path_graph, "graph_stats.json"), "w", encoding="utf-8") as fp:
        json.dump(stats, fp, sort_keys=True, indent=4, separators=(",", ": "))

    print("Finished.")
    return df_edges if copy else None


def pbg_train(dirname=None, pbg_params=None, output="model", auto_wd=True, save_wd=False):
    """Train PBG model with generated graph input.

    中文说明：
    - auto_wd=True 时按边规模估计权重衰减；
    - 训练前先把文本边表转换为 PBG 内部二进制格式；
    - 训练参数主体沿用 settings.pbg_params，保持与原算法接口兼容。
    """

    pbg_params = settings.pbg_params.copy() if pbg_params is None else pbg_params
    if not isinstance(pbg_params, dict):
        raise TypeError("`pbg_params` must be dict")

    if dirname is None:
        path_graph = Path(pbg_params["entity_path"]).parent.parent.as_posix()
    else:
        path_graph = os.path.join(settings.workdir, "pbg", dirname)

    pbg_params["checkpoint_path"] = os.path.join(path_graph, output)
    settings.pbg_params["checkpoint_path"] = pbg_params["checkpoint_path"]

    if auto_wd:
        n_edges = settings.graph_stats[os.path.basename(path_graph)]["n_edges"]
        if n_edges < 5e7:
            wd = np.around(0.013 * 2725781 / n_edges, decimals=6)
        else:
            wd = np.around(0.0004 * 59103481 / n_edges, decimals=6)
        print(f"Auto-estimated weight decay is {wd}")
        pbg_params["wd"] = wd
        if save_wd:
            settings.pbg_params["wd"] = wd
            print(f"`.settings.pbg_params['wd']` has been updated to {wd}")

    # PBG multiprocessing works better when each worker uses a single OpenMP thread.
    os.environ["OMP_NUM_THREADS"] = "1"

    loader = ConfigFileLoader()
    config = loader.load_config_simba(pbg_params)
    set_logging_verbosity(config.verbose)

    input_edge_paths = [Path(os.path.join(path_graph, "pbg_graph.txt"))]
    print("Converting input data ...")
    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    print("Starting training ...")
    train(attr.evolve(config, edge_paths=config.edge_paths), subprocess_init=subprocess_init)
    print("Finished")
