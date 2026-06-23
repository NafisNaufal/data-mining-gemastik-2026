#!/usr/bin/env python
"""C2 extension: Indonesian L1/EAI lineage + cross-lineage genomic transfer-gap.

TB Portals has ~zero L1/EAI even unpaired (7 specimens), and the Indonesian deployment
context is dominated by L1/EAI. We therefore call lineage + genotypic resistance on real
Indonesian WGS (NCBI SRA, via TB-Profiler; genomics-only -- no paired CXR exists for
Indonesia) and quantify how the teacher's *privileged genomic structure* transfers across
lineage/geography.

Two analyses:
  1. Distribution: Indonesian L1/EAI lineage fraction + rifampicin/MDR prevalence vs TB Portals.
  2. Transfer model: train a genomics-only rifampicin classifier on the EXACT teacher
     privileged-genomic feature set (lineage one-hot + co-resistance to the other 16 drugs +
     mutation burden; the rpoB/rifampicin determinant is EXCLUDED, identical leakage control
     to the teacher) on TB Portals L2/L4, then test in-distribution (TB Portals L2/L4 test)
     vs out-of-distribution (Indonesian L1). The AUROC gap measures cross-lineage transfer.

Run (gemastik env, after WGS pipeline + collate):
  PYTHONPATH=. python scripts/indonesia_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pusketb.config import load_config
from pusketb.data import genomics_features as gf

DRUGS = ["isoniazid", "rifampicin", "ethambutol", "pyrazinamide", "levofloxacin",
         "moxifloxacin", "amikacin", "kanamycin", "capreomycin", "streptomycin",
         "ethionamide", "linezolid", "bedaquiline", "clofazimine", "delamanid",
         "cycloserine", "para_aminosalicylic_acid"]
LIN_MAP = {f"lineage{i}": f"L{i}" for i in range(1, 7)}


def _drugs_of_variant(v: dict) -> set[str]:
    """Collect drug names a TB-Profiler dr_variant confers resistance to (schema-robust)."""
    out: set[str] = set()
    d = v.get("drugs") or v.get("gene_associated_drugs") or []
    if isinstance(d, list):
        for item in d:
            name = item.get("drug") if isinstance(item, dict) else item
            if name:
                out.add(str(name).strip().lower().replace("-", "_").replace(" ", "_"))
    elif isinstance(d, str):
        out.add(d.strip().lower().replace("-", "_").replace(" ", "_"))
    return out


def parse_indonesia(results_dir: Path, acc_tsv: Path,
                    min_depth: float = 10.0, min_mapped: float = 90.0) -> pd.DataFrame:
    prov = {r.run: (r.provenance, r.role) for r in
            pd.read_csv(acc_tsv, sep="\t").itertuples()}
    rows, dropped = [], 0
    for jf in sorted(results_dir.glob("*.results.json")):
        d = json.load(open(jf))
        sid = jf.name.replace(".results.json", "")
        qc = d.get("qc", {}) or {}
        depth = qc.get("target_median_depth") or qc.get("genome_median_depth") or 0
        mapped = qc.get("percent_reads_mapped") or qc.get("pct_reads_mapped") or 0
        if depth < min_depth or mapped < min_mapped:
            dropped += 1
            continue
        main = (d.get("main_lineage") or "").split(";")[0].strip()
        lg = "mixed" if ";" in (d.get("main_lineage") or "") else LIN_MAP.get(main, "other")
        resd = set()
        for v in d.get("dr_variants", []) or []:
            resd |= _drugs_of_variant(v)
        row = {"condition_id": sid,
               "main_lineage": main, "sub_lineage": d.get("sub_lineage", ""),
               "lineage_group": lg, "drtype": d.get("drtype", ""),
               "num_drug_resistant_variants": len(d.get("dr_variants", []) or []),
               "num_other_variants": len(d.get("other_variants", []) or []),
               "provenance": prov.get(sid, ("?", "?"))[0],
               "role": prov.get(sid, ("?", "?"))[1],
               "target_median_depth": depth, "percent_reads_mapped": mapped}
        for drug in DRUGS:
            row[f"res_{drug}"] = int(drug in resd)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rif_resistant"] = df["res_rifampicin"]
    df["mdr"] = ((df["res_rifampicin"] == 1) & (df["res_isoniazid"] == 1)).astype(int)
    print(f"parsed {len(df)} Indonesian samples (QC-dropped {dropped}; depth>={min_depth}, mapped>={min_mapped}%)")
    return df


def main() -> None:
    cfg = load_config()
    proc = Path(cfg.paths.processed_dir)
    wgs = Path("data/indonesia_wgs")
    indo = parse_indonesia(wgs / "results", wgs / "accessions.tsv")

    # ---- TB Portals condition table + train ids (reuse existing pipeline) -------------
    cond = pd.read_parquet(Path(cfg.paths.interim_dir) / "condition_labels.parquet")
    manifest = pd.read_parquet(proc / "manifest.parquet")
    train_ids = manifest.loc[manifest["split"] == "train", "condition_id"].unique().tolist()
    test_ids = set(manifest.loc[manifest["split"] == "test", "condition_id"].unique())

    # Fit the genomic feature spec exactly as the teacher does (rif target -> rpoB excluded).
    spec = gf.fit_spec(cond, cfg, train_ids, target="rif_resistant", include_clinical=False)
    feat_cols = spec.feature_names
    print(f"transfer feature set: {len(feat_cols)} cols (rpoB/rifampicin EXCLUDED): "
          f"{[c for c in feat_cols if c.startswith('res_')][:5]}...")

    # Ensure Indonesian frame has every column the spec expects.
    for c in ["num_drug_resistant_variants", "num_other_variants"]:
        indo[c] = pd.to_numeric(indo[c], errors="coerce").fillna(0)
    X_indo = gf.transform(indo, spec).set_index("condition_id")[feat_cols]
    X_tbp = gf.transform(cond, spec).set_index("condition_id")[feat_cols]

    y_tbp = cond.set_index("condition_id")["rif_resistant"]
    # Train on TB Portals L2/L4 train; in-dist test = TB Portals L2/L4 test.
    cond_idx = cond.set_index("condition_id")
    is_l24 = cond_idx["lineage_group"].isin(["L2", "L4"])
    tr_ids = [i for i in train_ids if i in cond_idx.index and is_l24.get(i, False)]
    te_ids = [i for i in test_ids if i in cond_idx.index and is_l24.get(i, False)]
    indo_l1 = indo[indo["lineage_group"] == "L1"]["condition_id"].tolist()

    Xtr, ytr = X_tbp.loc[tr_ids], y_tbp.loc[tr_ids]
    Xte, yte = X_tbp.loc[te_ids], y_tbp.loc[te_ids]
    indo_idx = indo.set_index("condition_id")

    # OOD evaluation cohorts inside the Indonesian set.
    def ood_ids(mask):
        return [i for i in indo_idx.index[mask].tolist() if i in X_indo.index]
    cohorts = {
        "indonesia_all": ood_ids(indo_idx["rif_resistant"].notna()),
        "indonesia_L2L4": ood_ids(indo_idx["lineage_group"].isin(["L2", "L4"])),
        "indonesia_L1": ood_ids(indo_idx["lineage_group"] == "L1"),
    }

    print(f"\nTRAIN TBP L2/L4: n={len(Xtr)} prev={ytr.mean():.3f} | "
          f"IN-DIST test L2/L4: n={len(Xte)} prev={yte.mean():.3f}")
    print("Indonesian rif prevalence by lineage:")
    for lg in ["L1", "L2", "L4"]:
        s = indo_idx[indo_idx["lineage_group"] == lg]["rif_resistant"]
        print(f"  {lg}: n={len(s)} rif_prev={s.mean():.3f} ({int(s.sum())} resistant)")

    results = {}
    for name, clf in [("logreg", make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced"))),
                      ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05))]:
        clf.fit(Xtr, ytr)
        a_in = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        rec = {"in_dist_L2L4": round(a_in, 4)}
        line = f"  {name:7}  in-dist L2/L4 AUROC={a_in:.4f}"
        for cname, ids in cohorts.items():
            y = indo_idx.loc[ids, "rif_resistant"]
            n_pos = int(y.sum())
            if y.nunique() > 1:
                a = roc_auc_score(y, clf.predict_proba(X_indo.loc[ids])[:, 1])
                rec[cname] = {"auroc": round(a, 4), "n": len(ids), "n_pos": n_pos}
                line += f" | {cname} AUROC={a:.4f}(n={len(ids)},pos={n_pos})"
            else:
                rec[cname] = {"auroc": None, "n": len(ids), "n_pos": n_pos,
                              "note": "degenerate label (single class)"}
                line += f" | {cname} n={len(ids)},pos={n_pos} [degenerate]"
        results[name] = rec
        print(line)

    # ---- distribution summary -------------------------------------------------------
    dist = {
        "indonesia": {
            "n_total": int(len(indo)),
            "lineage_counts": indo["lineage_group"].value_counts().to_dict(),
            "l1_eai_fraction": round((indo["lineage_group"] == "L1").mean(), 3),
            "rif_prevalence": round(indo["rif_resistant"].mean(), 3),
            "mdr_prevalence": round(indo["mdr"].mean(), 3),
            "by_provenance": indo.groupby("provenance")["lineage_group"]
                .value_counts().unstack(fill_value=0).to_dict(),
        },
        "tbportals": {
            "n_total": int(len(cond)),
            "lineage_counts": cond["lineage_group"].value_counts().to_dict(),
            "l1_eai_fraction": round((cond["lineage_group"] == "L1").mean(), 4),
            "rif_prevalence": round(cond["rif_resistant"].mean(), 3),
        },
        "transfer": results,
    }
    out = proc / "indonesia_transfer_results.json"
    json.dump(dist, open(out, "w"), indent=2, default=str)
    indo.to_parquet(wgs / "indonesia_cond.parquet")
    print(f"\nIndonesia lineage: {dist['indonesia']['lineage_counts']}")
    print(f"L1/EAI fraction: Indonesia {dist['indonesia']['l1_eai_fraction']} vs "
          f"TB Portals {dist['tbportals']['l1_eai_fraction']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
