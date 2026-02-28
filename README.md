# SIMBA (Minimal scATAC Project)

This repository is trimmed to one purpose:

- process **one scATAC AnnData** (`.h5ad`)
- run the SIMBA scATAC tutorial workflow end-to-end
- keep **kmer/motif** branch and downstream analysis

The retained workflow matches the tutorial chain:

1. `filter_peaks` / `filter_cells_atac` / `cal_qc_atac`
2. `pca` + `select_pcs` + `select_pcs_features`
3. `write_bed` and kmer/motif scanning (`R_scripts/scan_for_kmers_motifs.R`)
4. `gen_graph` + `pbg_train`
5. `read_embedding`
6. `umap` / `embed` / `compare_entities` / `query`
7. save all key AnnData outputs

## Installation

```bash
pip install -e .
```

## kmer/motif prerequisites

Install required R packages (example via conda):

```bash
conda install r-essentials r-optparse bioconductor-jaspar2020 bioconductor-biostrings bioconductor-tfbstools bioconductor-motifmatchr bioconductor-summarizedexperiment r-doparallel bioconductor-rhdf5 bioconductor-hdf5array
```

`scan_for_kmers_motifs.R` also needs `bedtools` available in PATH.

## Run pipeline

### Option A: use your own scATAC h5ad + precomputed kmer/motif h5

```bash
python scripts/run_scatac_tutorial_pipeline.py \
  --input-h5ad /path/to/your_scatac.h5ad \
  --kmer-h5 /path/to/freq_kmer.h5 \
  --motif-h5 /path/to/freq_motif.h5 \
  --workdir result_simba_atacseq
```

### Option B: run R scan inside the pipeline

```bash
python scripts/run_scatac_tutorial_pipeline.py \
  --input-h5ad /path/to/your_scatac.h5ad \
  --run-scan \
  --genome-fasta /path/to/hg19.fa \
  --species "Homo sapiens" \
  --workdir result_simba_atacseq
```

### Option C: run tutorial example dataset

```bash
python scripts/run_scatac_tutorial_pipeline.py \
  --use-example-data \
  --run-scan \
  --genome-fasta /path/to/hg19.fa \
  --species "Homo sapiens" \
  --workdir result_simba_atacseq
```

## Main outputs

- `adata_CP.h5ad`, `adata_PK.h5ad`, `adata_PM.h5ad`
- `adata_C.h5ad`, `adata_P.h5ad`, `adata_K.h5ad`, `adata_M.h5ad`, `adata_all.h5ad`
- `adata_cmp_CM.h5ad`, `adata_cmp_CK.h5ad`
- figures under `<workdir>/figures`
- `run_manifest.json` (runtime config + stage timings)

## Project layout

The pipeline logic now lives inside `simba` itself:

- `simba/pipeline.py`: end-to-end scATAC pipeline (prep/train/analyze/save)
- `simba/tools/_pbg.py`: graph table construction + PBG training
- `simba/preprocessing/*`: ATAC preprocessing (QC/PCA/select top features)
- `simba/plotting/*`: compact plotting utilities for QC and post-analysis
- `scripts/run_scatac_tutorial_pipeline.py`: thin executable wrapper

## Repository scope

Removed content includes docs, CI, tests, RNA/integration/gene-score modules, and unrelated datasets.
Only scATAC + kmer/motif tutorial functionality is kept.
