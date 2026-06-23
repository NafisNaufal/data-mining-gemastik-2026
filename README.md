# PuskeTB — cross-modal distillation for drug-resistant TB triage

GEMASTIK 2026, Penambangan Data.
This README documents the **data pipeline** (the first build) and the best-practice
decisions taken on the way.

## Environment

Everything runs in the `gemastik` conda env:

```bash
conda activate gemastik          # Python 3.10
pip install -r requirements.txt  # torch/torchvision from the cu121 index (see file header)
```

Hardware available: 3× NVIDIA L40 (46 GB), CUDA 12.1.

## What the data actually is (measured, not assumed)

Raw TB Portals drops live under `tbportals/` (DUA-protected, **git-ignored, never committed**).

| Table | Rows | Key |
|---|---|---|
| Genomics (`TB_Portals_Genomics_March_2026.csv`) | 9,333 specimens | `condition_id` |
| CXR metadata (`TB_Portals_CXRs_March_2026.csv`) | 18,142 series | `condition_id` |
| Sequenced DST (phenotypic) | 44,364 rows | `condition_id` |

- **Paired X-ray + genomics subset ≈ 6,781 patients** — the load-bearing core (C1/C2).
- DICOMs are multi-per-patient, ~12 MB each, ~100 GB+ across 20 zip parts (still zipped).
- Linkage: the CXR table's `series_instance_content_url` is identical to the `file`
  column in each image-part manifest → maps every series to its `*_NN_of_20.zip`.

### The lineage reality (drives the C2 framing)

Paired-subset lineage: **L2 ≈ 3,925, L4 ≈ 3,017, L3 ≈ 12, L1/EAI = 7.** TB Portals is
Eastern-European-dominant; there is **no Indonesian data** here, and the Indonesian WGS
sources are genomics-only (no paired X-rays). So:

> **C2 decision:** demonstrate the lineage-stratified evaluation *method* on **L2 vs L4**
> (both abundant), and treat **L1/EAI as an external transfer-gap analysis** using
> Indonesian genomics + literature. Honest, feasible, still a first-of-kind for TB-DR ML.

## Pipeline

```bash
# 1) Build the paired manifest + patient-grouped stratified splits (CSV-only, ~seconds)
python scripts/build_manifest.py
#    -> data/processed/manifest.parquet, split_summary.csv, lineage_label_crosstab.csv

# 2) Decode DICOMs -> normalized PNGs (resumable, shardable by zip part)
python scripts/extract_images.py --limit 50          # smoke test
python scripts/extract_images.py --part 01_of_20     # one part (run several in parallel)

# 3) Cache frozen RAD-DINO embeddings for fast prototyping
python scripts/cache_embeddings.py --split train
```

## Best-practice decisions

- **Target = rifampicin resistance (binary).** It is the GeneXpert/Xpert MTB-RIF analogue,
  which makes the cost-sensitive triage story (C3) clinically coherent. Derived genotype-first
  from the curated `rifampicin` mutation column (`-` = susceptible), cross-checked against the
  curated `drug_resistance_type`; disagreements are flagged (`rif_label_disagreement`).
  All 17 drugs are binarized and carried, so the target can change without re-deriving.
- **Patient-grouped splits.** Split on `condition_id`, never on images, so a patient's
  multiple CXRs cannot leak across train/test. Stratified on `lineage_group × target`.
- **Dedicated calibration split** (70/10/20 train/calib/test) reserved up front for
  conformal prediction (C4) — calibration must be disjoint from train *and* test.
- **Encoder-frozen first, end-to-end later.** The pipeline emits decoded CXR tensors
  (so end-to-end RAD-DINO fine-tuning is fully supported) *and* an embedding cache (so the
  teacher, distilled student, ranking loss, and conformal layer can be prototyped cheaply).
  Staged path gives the better-result end-to-end option without blocking on the heavy
  DICOM pipeline.
- **Modality filter:** keep `CR`/`DX`/`XC` (radiographs + photographed films), drop `XA`
  (angiography). `cxr_outlier` is carried as a column, not silently dropped.

## Layout

```
configs/data.yaml          # all paths, label/lineage/split/decoding settings
pusketb/
  config.py                # config loader (resolves paths to project root)
  data/
    sources.py             # read raw CSVs straight from the DUA zips
    labels.py              # genomics -> condition-level binary labels + lineage
    manifest.py            # join imaging <-> labels (paired subset) + zip mapping
    splits.py              # patient-grouped, stratified train/calib/test
    dicom_extract.py       # DICOM -> normalized PNG (windowing, MONOCHROME1, resize)
  encoders/rad_dino.py     # frozen RAD-DINO feature extractor
scripts/                   # build_manifest / extract_images / cache_embeddings
```

## Data governance

TB Portals data is under a Data Use Agreement that prohibits redistribution. `tbportals/`
and `data/` are git-ignored. Do not commit patient data, DICOMs, embeddings, or field-study
material with identifiers.
