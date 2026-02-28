"""End-to-end scATAC pipeline built on top of SIMBA + PBG.

中文说明：
该模块用于把教程中的完整流程串成一条清晰主链路：
输入 h5ad -> 预处理（CP/PK/PM）-> 构图与 PBG 训练 -> 下游分析 -> 结果保存。
设计目标是“步骤显式、入口单一、可追踪复现”。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import datasets, plotting as pl, preprocessing as pp, tools as tl
from ._settings import settings
from .readwrite import load_graph_stats, load_pbg_config, read_embedding, read_h5ad, read_hdf, write_bed


@dataclass
class ScATACConfig:
    input_h5ad: str | None = None
    use_example_data: bool = False
    workdir: str = "result_simba_atacseq"

    min_peak_cells: int = 3
    min_cell_peaks: int = 100
    pca_components: int = 50
    n_pcs: int = 30

    kmer_h5: str | None = None
    motif_h5: str | None = None
    run_scan: bool = False
    scan_script: str = os.path.join("R_scripts", "scan_for_kmers_motifs.R")
    genome_fasta: str | None = None
    species: str | None = None

    pbg_epochs: int = 10
    umap_neighbors: int = 15
    query_entity: str | None = None


@dataclass
class PrepArtifacts:
    adata_CP: Any
    adata_PK: Any
    adata_PM: Any


@dataclass
class AnalyzeArtifacts:
    adata_C: Any
    adata_P: Any
    adata_K: Any
    adata_M: Any
    adata_all: Any
    adata_cmp_CM: Any
    adata_cmp_CK: Any


def _decode_index(values):
    """将 AnnData 索引统一转成字符串，兼容 bytes/str 混合输入。"""
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _normalize_indices(adata):
    """规范化 obs/var 的索引类型，避免后续 join/loc 因类型不一致失败。"""
    adata.obs.index = _decode_index(adata.obs.index)
    adata.var.index = _decode_index(adata.var.index)


def _get_entity(dict_adata, prefix):
    """按实体前缀获取 embedding，对前缀匹配提供兜底逻辑。"""
    if prefix in dict_adata:
        return dict_adata[prefix]
    matches = [key for key in dict_adata if key.startswith(prefix)]
    if not matches:
        raise KeyError(f"Entity prefix '{prefix}' is not present in embeddings: {list(dict_adata.keys())}")
    return dict_adata[matches[0]]


def _preflight(config: ScATACConfig) -> None:
    """运行前校验。

    中文说明：
    1) 保证输入数据来源明确（本地 h5ad 或示例数据）；
    2) 若启用 kmer/motif 扫描，校验 R 脚本、基因组 fasta、species 和 Rscript 环境。
    """
    if not config.use_example_data and not config.input_h5ad:
        raise ValueError("Specify --input-h5ad, or set --use-example-data.")
    if config.input_h5ad and not os.path.exists(config.input_h5ad):
        raise FileNotFoundError(f"Input AnnData not found: {config.input_h5ad}")

    if not config.run_scan:
        return

    if not config.genome_fasta:
        raise ValueError("`--genome-fasta` is required when `--run-scan` is enabled.")
    if not os.path.exists(config.genome_fasta):
        raise FileNotFoundError(f"Genome fasta not found: {config.genome_fasta}")
    if not config.species:
        raise ValueError("`--species` is required when `--run-scan` is enabled.")
    if not os.path.exists(config.scan_script):
        raise FileNotFoundError(f"Scan script not found: {config.scan_script}")
    if shutil.which("Rscript") is None:
        raise EnvironmentError("`Rscript` not found in PATH.")


def _load_cp(config: ScATACConfig):
    """加载 cell-peak 矩阵（CP）。"""
    if config.use_example_data:
        print("Loading tutorial example dataset atac_buenrostro2018 ...")
        return datasets.atac_buenrostro2018()

    print(f"Loading input AnnData: {config.input_h5ad}")
    return read_h5ad(config.input_h5ad)


def _run_kmer_motif_scan(config: ScATACConfig, peaks_bed: str) -> None:
    """调用 R 脚本从 peaks.bed 扫描 kmer/motif。"""
    cmd = [
        "Rscript",
        config.scan_script,
        "-i",
        peaks_bed,
        "-g",
        config.genome_fasta,
        "-s",
        config.species,
    ]
    print("Running kmer/motif scan:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _resolve_kmer_motif_paths(config: ScATACConfig) -> tuple[str, str]:
    """解析 kmer/motif 输入路径。

    中文说明：
    - 优先使用显式传入的 h5 路径；
    - 未传入时使用 workdir 下教程默认输出路径；
    - 若 run_scan=True，会先触发 R 扫描再校验产物是否存在。
    """
    default_kmer = os.path.join(config.workdir, "output_kmers_motifs", "freq_kmer.h5")
    default_motif = os.path.join(config.workdir, "output_kmers_motifs", "freq_motif.h5")
    kmer_h5 = config.kmer_h5 or default_kmer
    motif_h5 = config.motif_h5 or default_motif

    if config.run_scan:
        peaks_bed = os.path.join(config.workdir, "peaks.bed")
        _run_kmer_motif_scan(config, peaks_bed)

    if not os.path.exists(kmer_h5) or not os.path.exists(motif_h5):
        raise FileNotFoundError(
            "kmer/motif matrix files are missing. Provide --kmer-h5 and --motif-h5, "
            "or run with --run-scan, --genome-fasta and --species."
        )

    return kmer_h5, motif_h5


def prep_stage(config: ScATACConfig) -> PrepArtifacts:
    """预处理阶段：构建 CP/PK/PM 三类输入矩阵。

    中文说明：
    该阶段负责所有“喂给 PBG 前”的数据准备：
    - CP 的 QC、过滤、PCA 与 top_pcs 特征筛选；
    - 从 peaks 读取 PK/PM 后进行二值化和特征筛选。
    """
    adata_CP = _load_cp(config)

    print("Preprocessing scATAC matrix ...")
    pp.filter_peaks(adata_CP, min_n_cells=config.min_peak_cells)
    pp.cal_qc_atac(adata_CP)
    pl.violin(adata_CP, list_obs=["n_counts", "n_peaks", "pct_peaks"], list_var=["n_cells"], fig_size=(3, 3))
    pl.hist(
        adata_CP, 
        list_obs=["n_counts", "n_peaks", "pct_peaks"],
        list_var=["n_cells"],
        log=True,
        fig_size=(3, 3),
    )
    pp.filter_cells_atac(adata_CP, min_n_peaks=config.min_cell_peaks)

    print("Selecting top PCs-associated peaks ...")
    pp.pca(adata_CP, n_components=config.pca_components)
    pl.pca_variance_ratio(adata_CP, show_cutoff=True)
    if config.n_pcs > 0:
        pp.select_pcs(adata_CP, n_pcs=config.n_pcs)
    else:
        pp.select_pcs(adata_CP)
    pp.select_pcs_features(adata_CP)
    pl.pcs_features(adata_CP, fig_ncol=10)

    print("Writing peaks.bed ...")
    write_bed(adata_CP, use_top_pcs=True)

    kmer_h5, motif_h5 = _resolve_kmer_motif_paths(config)

    print("Loading kmer/motif matrices ...")
    adata_PK = read_hdf(kmer_h5, "mat")
    adata_PM = read_hdf(motif_h5, "mat")
    _normalize_indices(adata_PK)
    _normalize_indices(adata_PM)

    pp.binarize(adata_PK)
    pp.binarize(adata_PM)

    print("Selecting top PCs-associated kmers/motifs ...")
    pp.pca(adata_PK, n_components=config.pca_components)
    pp.pca(adata_PM, n_components=config.pca_components)
    pp.select_pcs_features(adata_PK, min_elbow=max(1, adata_PK.shape[1] // 5), S=5)
    pp.select_pcs_features(adata_PM, min_elbow=max(1, adata_PM.shape[1] // 5), S=5)
    pl.pcs_features(adata_PK, fig_ncol=10)
    pl.pcs_features(adata_PM, fig_ncol=10)

    return PrepArtifacts(adata_CP=adata_CP, adata_PK=adata_PK, adata_PM=adata_PM)


def train_stage(config: ScATACConfig, prep: PrepArtifacts) -> None:
    """训练阶段：生成关系图并启动 PBG 训练。"""
    print("Generating graph and training PBG ...")
    tl.gen_graph(
        list_CP=[prep.adata_CP],
        list_PK=[prep.adata_PK],
        list_PM=[prep.adata_PM],
        copy=False,
        use_top_pcs=True,
        dirname="graph0",
    )

    pbg_params = settings.pbg_params.copy()
    pbg_params["num_epochs"] = config.pbg_epochs
    tl.pbg_train(pbg_params=pbg_params, auto_wd=True, save_wd=True, output="model")

    load_graph_stats()
    load_pbg_config()
    pl.pbg_metrics(fig_ncol=1)


def _set_entity_annotations(adata_all, adata_C, adata_P, adata_K, adata_M):
    """统一写入 entity_anno，便于下游 UMAP 与 query 可视化。"""
    adata_all.obs["entity_anno"] = ""
    if "celltype" in adata_C.obs:
        adata_all.obs.loc[adata_C.obs_names, "entity_anno"] = adata_C.obs["celltype"].tolist()
    else:
        adata_all.obs.loc[adata_C.obs_names, "entity_anno"] = "cell"

    adata_all.obs.loc[adata_P.obs_names, "entity_anno"] = "peak"
    adata_all.obs.loc[adata_K.obs_names, "entity_anno"] = "kmer"
    adata_all.obs.loc[adata_M.obs_names, "entity_anno"] = "motif"


def analyze_stage(config: ScATACConfig, prep: PrepArtifacts) -> AnalyzeArtifacts:
    """下游分析阶段。

    中文说明：
    - 读取训练得到的 C/P/K/M embedding；
    - 先对 C 做 UMAP，再把 M/K/P 投影到同一坐标；
    - 计算 motif/kmer 对 cell 的实体指标，并可选执行 query 演示。
    """
    print("Reading embeddings and running downstream analysis ...")
    dict_adata = read_embedding()
    adata_C = _get_entity(dict_adata, "C")
    adata_P = _get_entity(dict_adata, "P")
    adata_K = _get_entity(dict_adata, "K")
    adata_M = _get_entity(dict_adata, "M")

    if "celltype" in prep.adata_CP.obs:
        adata_C.obs["celltype"] = prep.adata_CP[adata_C.obs_names, :].obs["celltype"].copy()

    tl.umap(adata_C, n_neighbors=config.umap_neighbors, n_components=2)
    if "celltype" in adata_C.obs:
        pl.umap(adata_C, color=["celltype"], fig_size=(4.5, 4), drawing_order="random")
    else:
        pl.umap(adata_C, fig_size=(4.5, 4), drawing_order="random")

    adata_all = tl.embed(adata_ref=adata_C, list_adata_query=[adata_M, adata_K, adata_P])
    _set_entity_annotations(adata_all, adata_C, adata_P, adata_K, adata_M)

    tl.umap(adata_all, n_neighbors=config.umap_neighbors, n_components=2)
    pl.umap(
        adata_all[::-1, :],
        color=["entity_anno"],
        fig_size=(6.5, 5.5),
        drawing_order="original",
        show_texts=False,
    )

    adata_cmp_CM = tl.compare_entities(adata_ref=adata_C, adata_query=adata_M)
    adata_cmp_CK = tl.compare_entities(adata_ref=adata_C, adata_query=adata_K)

    pl.entity_metrics(
        adata_cmp_CM,
        x="max",
        y="gini",
        show_texts=True,
        show_cutoff=True,
        show_contour=True,
        cutoff_x=1.8,
        cutoff_y=0.5,
        c="#92ba79",
        save_fig=False,
    )
    pl.entity_metrics(
        adata_cmp_CK,
        x="max",
        y="gini",
        show_texts=True,
        show_cutoff=True,
        show_contour=True,
        cutoff_x=1.2,
        cutoff_y=0.32,
        c="#94b1b7",
        save_fig=False,
    )

    if config.query_entity:
        print(f"Running query demo around entity: {config.query_entity}")
        query_result = tl.query(adata_all, entity=[config.query_entity], obsm="X_umap", k=50)
        print(query_result.head(10))
        pl.query(
            adata_all,
            obsm="X_umap",
            color=["entity_anno"],
            show_texts=False,
            alpha=0.9,
            alpha_bg=0.1,
            fig_size=(6.5, 5.5),
        )

    return AnalyzeArtifacts(
        adata_C=adata_C,
        adata_P=adata_P,
        adata_K=adata_K,
        adata_M=adata_M,
        adata_all=adata_all,
        adata_cmp_CM=adata_cmp_CM,
        adata_cmp_CK=adata_cmp_CK,
    )


def save_stage(config: ScATACConfig, prep: PrepArtifacts, out: AnalyzeArtifacts) -> None:
    """统一保存各阶段关键 AnnData 结果，便于复盘。"""
    print("Saving output AnnData objects ...")
    outputs = {
        "adata_CP.h5ad": prep.adata_CP,
        "adata_PK.h5ad": prep.adata_PK,
        "adata_PM.h5ad": prep.adata_PM,
        "adata_C.h5ad": out.adata_C,
        "adata_P.h5ad": out.adata_P,
        "adata_K.h5ad": out.adata_K,
        "adata_M.h5ad": out.adata_M,
        "adata_all.h5ad": out.adata_all,
        "adata_cmp_CM.h5ad": out.adata_cmp_CM,
        "adata_cmp_CK.h5ad": out.adata_cmp_CK,
    }
    for filename, adata in outputs.items():
        adata.write(os.path.join(config.workdir, filename))


def run_scatac_pipeline(config: ScATACConfig) -> dict[str, float]:
    """一键执行完整流程，并输出阶段耗时清单。"""
    #检查一下config完不完整
    _preflight(config)
    os.makedirs(config.workdir, exist_ok=True)

    settings.set_workdir(config.workdir)
    settings.set_figure_params(dpi=80, style="white", fig_size=[5, 5], rc={"image.cmap": "viridis"})

    stage_times: dict[str, float] = {}

    # 分阶段计时，方便定位耗时瓶颈（prep/train/analyze）。
    t0 = time.time()
    prep = prep_stage(config)
    stage_times["prep"] = round(time.time() - t0, 3)

    t0 = time.time()
    train_stage(config, prep)
    stage_times["train"] = round(time.time() - t0, 3)

    t0 = time.time()
    out = analyze_stage(config, prep)
    save_stage(config, prep, out)
    stage_times["analyze"] = round(time.time() - t0, 3)

    manifest_path = os.path.join(config.workdir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "workdir": str(Path(config.workdir).resolve()),
                "config": asdict(config),
                "stage_times_sec": stage_times,
            },
            fp,
            indent=2,
            sort_keys=True,
        )
    return stage_times


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal scATAC SIMBA workflow focused on PBG.")
    parser.add_argument("--input-h5ad", default=None, help="Path to input scATAC AnnData (.h5ad).")
    parser.add_argument("--use-example-data", action="store_true", help="Use si.datasets.atac_buenrostro2018().")
    parser.add_argument("--workdir", default="result_simba_atacseq", help="Output working directory.")

    parser.add_argument("--min-peak-cells", type=int, default=3, help="Minimum number of cells per peak.")
    parser.add_argument("--min-cell-peaks", type=int, default=100, help="Minimum number of peaks per cell.")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components.")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of PCs kept for peak feature selection.")

    parser.add_argument("--kmer-h5", default=None, help="Path to freq_kmer.h5 (key: mat).")
    parser.add_argument("--motif-h5", default=None, help="Path to freq_motif.h5 (key: mat).")
    parser.add_argument("--run-scan", action="store_true", help="Run R scan_for_kmers_motifs.R after writing peaks.bed.")
    parser.add_argument("--scan-script", default=os.path.join("R_scripts", "scan_for_kmers_motifs.R"))
    parser.add_argument("--genome-fasta", default=None, help="Path to genome fasta for R scan.")
    parser.add_argument("--species", default=None, help="Species name for R scan, e.g. 'Homo sapiens'.")

    parser.add_argument("--pbg-epochs", type=int, default=10, help="PBG training epochs.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="Neighbors for UMAP.")
    parser.add_argument("--query-entity", default=None, help="Optional entity name for query demo.")
    return parser


def parse_args() -> ScATACConfig:
    args = build_parser().parse_args()
    return ScATACConfig(
        input_h5ad=args.input_h5ad,
        use_example_data=args.use_example_data,
        workdir=args.workdir,
        min_peak_cells=args.min_peak_cells,
        min_cell_peaks=args.min_cell_peaks,
        pca_components=args.pca_components,
        n_pcs=args.n_pcs,
        kmer_h5=args.kmer_h5,
        motif_h5=args.motif_h5,
        run_scan=args.run_scan,
        scan_script=args.scan_script,
        genome_fasta=args.genome_fasta,
        species=args.species,
        pbg_epochs=args.pbg_epochs,
        umap_neighbors=args.umap_neighbors,
        query_entity=args.query_entity,
    )


def main() -> None:
    config = parse_args()
    times = run_scatac_pipeline(config)
    print("Done. Results are in:", config.workdir)
    print("Stage time summary (sec):", times)
