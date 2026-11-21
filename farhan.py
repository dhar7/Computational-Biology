#!/usr/bin/env python3
"""Farhan's part: curate data, remove overlap, create peptide-grouped folds."""

from __future__ import annotations

from bootstrap_runtime import prefer_project_venv

prefer_project_venv()

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from study_common import (
    CANONICAL_ALLELES,
    Progress,
    atomic_json,
    ensure_dir,
    get_processed_paths,
    is_valid_aa,
    load_config,
    normalize_aa,
    normalize_hla,
    read_tsv,
    resolve_path,
    stable_pair_id,
    write_tsv,
)


def assign_peptide_folds(rows: list[dict[str, str]], folds: int, seed: int) -> dict[str, int]:
    import random

    peptide_sizes = collections.Counter(row["peptide"] for row in rows)
    peptides = list(peptide_sizes)
    if len(peptides) < folds:
        raise ValueError(f"Need at least {folds} unique peptides to make {folds} grouped folds; found {len(peptides)}")
    rng = random.Random(seed)
    tie_order = {peptide: rng.random() for peptide in peptides}
    peptides.sort(key=lambda peptide: (-peptide_sizes[peptide], tie_order[peptide]))
    totals = [0] * folds
    assignment: dict[str, int] = {}
    for peptide in peptides:
        minimum = min(totals)
        choices = [index for index, total in enumerate(totals) if total == minimum]
        fold = rng.choice(choices)
        assignment[peptide] = fold
        totals[fold] += peptide_sizes[peptide]
    return assignment


def run(config: dict, config_root: Path) -> None:
    data = config["data"]
    paths = get_processed_paths(config, config_root)
    delimiter = data.get("delimiter", "\t")
    positive_path = resolve_path(config_root, data["positives_path"])
    repertoire_path = resolve_path(config_root, data["repertoire_path"])
    missing_inputs = [path for path in (positive_path, repertoire_path) if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Required input file(s) missing: " + ", ".join(str(path) for path in missing_inputs)
        )
    output_root = resolve_path(config_root, config.get("output_dir", "outputs")) / "farhan"
    progress = Progress("farhan", [
        ("Read and validate source tables", 15),
        ("Normalize and deduplicate positives", 25),
        ("Clean repertoire and remove binder overlap", 20),
        ("Assign peptide-grouped folds and check leakage", 25),
        ("Write processed tables and manifest", 15),
    ], output_root, config.get("progress_initial_estimates_minutes", {}).get("farhan", 10))

    raw_positive_rows = read_tsv(positive_path, delimiter)
    raw_repertoire_rows = read_tsv(repertoire_path, delimiter)
    positive_columns = data.get("positive_columns", {})
    repertoire_columns = data.get("repertoire_columns", {})
    for canonical in ("tcr_beta", "peptide", "hla"):
        column = positive_columns.get(canonical, canonical)
        if not raw_positive_rows or column not in raw_positive_rows[0]:
            raise ValueError(f"Positive input must include column {column!r} for {canonical}")
    rep_tcr_col = repertoire_columns.get("tcr_beta", "tcr_beta")
    if not raw_repertoire_rows or rep_tcr_col not in raw_repertoire_rows[0]:
        raise ValueError(f"Repertoire input must include column {rep_tcr_col!r} for tcr_beta")
    progress.tick(0, 1.0, f"read {len(raw_positive_rows):,} positive rows and {len(raw_repertoire_rows):,} repertoire rows")

    targets = {normalize_hla(allele) for allele in data.get("target_alleles", sorted(CANONICAL_ALLELES))}
    if not targets or "" in targets:
        raise ValueError("Every target_alleles entry must be a supported HLA allele")
    if targets != set(CANONICAL_ALLELES):
        raise ValueError("This study implementation is fixed to HLA-A*02:01 and HLA-B*40:01 as specified in the proposal")
    rejected = collections.Counter()
    source_column = data.get("positive_columns", {}).get("source", "source")
    raw_source_counts: collections.Counter[str] = collections.Counter()
    for raw in raw_positive_rows:
        raw_source = str(raw.get(source_column, "unknown")).strip() or "unknown"
        for source_name in raw_source.split(";"):
            raw_source_counts[source_name.strip() or "unknown"] += 1

    deduplicated: dict[tuple[str, str, str], dict[str, str]] = {}
    for index, raw in enumerate(raw_positive_rows):
        hla = normalize_hla(raw.get(positive_columns.get("hla", "hla"), ""))
        if hla not in targets:
            rejected["unsupported_or_invalid_hla"] += 1
            continue
        peptide = normalize_aa(raw.get(positive_columns.get("peptide", "peptide"), ""))
        tcr = normalize_aa(raw.get(positive_columns.get("tcr_beta", "tcr_beta"), ""))
        if not is_valid_aa(peptide, min_length=7):
            rejected["invalid_peptide"] += 1
            continue
        if not is_valid_aa(tcr, min_length=3):
            rejected["invalid_tcr_beta"] += 1
            continue
        source = str(raw.get(source_column, "unknown")).strip() or "unknown"
        source = source.replace("\t", " ")
        key = (hla, peptide, tcr)
        if key in deduplicated:
            prior = set(deduplicated[key]["source"].split(";"))
            prior.add(source)
            deduplicated[key]["source"] = ";".join(sorted(prior))
            rejected["duplicate_positive_pair"] += 1
        else:
            deduplicated[key] = {
                "pair_id": stable_pair_id(hla, peptide, tcr),
                "tcr_beta": tcr,
                "peptide": peptide,
                "hla": hla,
                "source": source,
            }
        if (index + 1) % 10000 == 0:
            progress.tick(1, (index + 1) / len(raw_positive_rows), f"checked {index + 1:,} positive rows")
    positives = list(deduplicated.values())
    if not positives:
        raise ValueError("No valid positives remain after allele and sequence filtering")
    progress.tick(1, 1.0, f"retained {len(positives):,} unique positive pairs")

    repertoire_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for index, raw in enumerate(raw_repertoire_rows):
        tcr = normalize_aa(raw.get(rep_tcr_col, ""))
        if not is_valid_aa(tcr, min_length=3):
            rejected["invalid_repertoire_tcr"] += 1
            continue
        hla_col = repertoire_columns.get("hla", "hla")
        allele = normalize_hla(raw.get(hla_col, "")) if hla_col in raw else ""
        if allele and allele not in targets:
            rejected["unsupported_repertoire_hla"] += 1
            continue
        repertoire_by_key.setdefault((tcr, allele), {"tcr_beta": tcr, "hla": allele})
        if (index + 1) % 10000 == 0:
            progress.tick(2, (index + 1) / len(raw_repertoire_rows), f"checked {index + 1:,} repertoire rows")

    positive_tcrs = {row["tcr_beta"] for row in positives}
    before_overlap = len(repertoire_by_key)
    repertoire = [
        row for (tcr, allele), row in sorted(repertoire_by_key.items())
        if tcr not in positive_tcrs
    ]
    rejected["repertoire_rows_overlapping_any_positive_tcr"] = before_overlap - len(repertoire)
    if not repertoire:
        raise ValueError("No background repertoire TCRs remain after removing known binders")
    progress.tick(2, 1.0, f"retained {len({row['tcr_beta'] for row in repertoire}):,} unique background TCRs; removed {before_overlap - len(repertoire):,} overlapping repertoire rows")

    n_folds = int(data.get("folds", 5))
    fold_map = assign_peptide_folds(positives, n_folds, int(data.get("fold_seed", 1601)))
    for row in positives:
        row["fold"] = str(fold_map[row["peptide"]])
    peptide_folds: dict[str, set[int]] = collections.defaultdict(set)
    for row in positives:
        peptide_folds[row["peptide"]].add(int(row["fold"]))
    leaked_peptides = [peptide for peptide, assigned in peptide_folds.items() if len(assigned) != 1]
    if leaked_peptides:
        raise AssertionError(f"Peptides crossed fold boundaries: {leaked_peptides[:5]}")
    fold_counts = collections.Counter(int(row["fold"]) for row in positives)
    if set(fold_counts) != set(range(n_folds)):
        raise AssertionError("At least one fold is empty")
    progress.tick(3, 1.0, "all peptides occur in exactly one fold; five folds are populated")

    source_counts: dict[str, int] = collections.Counter()
    for row in positives:
        for source in row["source"].split(";"):
            source_counts[source] += 1
    expected = data.get("expected_source_counts", {})
    expected_status = {
        source: {
            "expected_raw_rows": int(count),
            "observed_raw_rows": int(raw_source_counts.get(source, 0)),
            "matches": int(raw_source_counts.get(source, 0)) == int(count),
        }
        for source, count in expected.items()
    }
    # Background repertoire rows have no peptide field: remove and validate
    # binder overlap at the TCR sequence level across all known positives.
    if any(row["tcr_beta"] in positive_tcrs for row in repertoire):
        raise AssertionError("A known positive TCR remains in the background repertoire")

    ensure_dir(paths["dir"])
    ensure_dir(output_root)
    write_tsv(paths["positives"], positives)
    write_tsv(paths["repertoire"], repertoire)
    write_tsv(paths["dir"] / "folds.tsv", (
        {"pair_id": row["pair_id"], "peptide": row["peptide"], "hla": row["hla"], "fold": row["fold"]}
        for row in positives
    ))
    manifest = {
        "proposal": "TCR-pMHC negative construction study",
        "created_unix": time.time(),
        "source_files": {"positives": str(positive_path), "repertoire": str(repertoire_path)},
        "raw_rows": {"positives": len(raw_positive_rows), "repertoire": len(raw_repertoire_rows)},
        "retained_rows": {
            "positives": len(positives),
            "background_repertoire_rows": len(repertoire),
            "unique_background_tcr_sequences": len({row["tcr_beta"] for row in repertoire}),
        },
        "positive_source_counts": dict(sorted(source_counts.items())),
        "raw_positive_source_counts": dict(sorted(raw_source_counts.items())),
        "expected_source_count_check": expected_status,
        "target_alleles": sorted(targets),
        "folds": n_folds,
        "fold_seed": int(data.get("fold_seed", 1601)),
        "positive_rows_per_fold": {str(fold): fold_counts[fold] for fold in range(n_folds)},
        "unique_peptides": len(peptide_folds),
        "rejected_or_collapsed_rows": dict(rejected),
        "leakage_checks": {
            "peptide_grouped": True,
            "peptides_crossing_folds": 0,
            "known_positive_tcrs_in_background_repertoire": 0,
        },
        "notes": [
            "Expected source counts are reported, not enforced, because source tables may already be filtered or deduplicated.",
            "TCR and peptide sequences are uppercased and whitespace is removed; CDR3 anchor residues are retained.",
        ],
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(output_root / "manifest.json", manifest)
    progress.tick(4, 1.0, f"wrote processed data to {paths['dir']}")
    progress.tick(5, 1.0, "complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farhan", action="store_true", help="run Farhan's data pipeline")
    parser.add_argument("--config", default="study_config.json", help="path to study_config.json")
    args = parser.parse_args()
    if not args.farhan:
        parser.error("select this part with --farhan")
    config, root = load_config(args.config)
    try:
        run(config, root)
    except FileNotFoundError as exc:
        print(f"Input setup incomplete: {exc}", file=sys.stderr)
        print(
            "Add the project positive-pair TSV and background repertoire TSV at the paths in study_config.json.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
