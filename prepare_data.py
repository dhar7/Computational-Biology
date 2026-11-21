#!/usr/bin/env python3
"""Download and prepare the input tables required by farhan.py.

This uses the IMMREP25 positive training files named in the proposal and a
public 10x Genomics healthy-donor TCR repertoire. It uses only Python's
standard library; it does not install packages or run any study analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
IMMREP_ROOT = RAW_ROOT / "immrep25"
TENX_ROOT = RAW_ROOT / "10x"

IMMREP_FILES = {
    "IEDB": {
        "filename": "iedb_positives.csv",
        "url": "https://raw.githubusercontent.com/Ajaingithub/pMHC_TCR_specificity/main/data/iedb_positives.csv",
        "expected_rows": 9652,
        # IEDB CDR3b omits the conserved leading C; extended preserves anchors.
        "cdr3_column": "CDR3b_extended",
    },
    "VDJdb": {
        "filename": "vdjdb_positives.csv",
        "url": "https://raw.githubusercontent.com/Ajaingithub/pMHC_TCR_specificity/main/data/vdjdb_positives.csv",
        "expected_rows": 30274,
        # The supplied VDJdb CDR3b is already junction form; _extended adds
        # another C/F to many rows, so use CDR3b to avoid double anchors.
        "cdr3_column": "CDR3b",
    },
}

TENX_FILENAME = "tcr_all_contig_annotations.csv"
TENX_URL = (
    "https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/"
    "sc5p_v2_hs_PBMC_10k_multi_5gex_5fb_b_t/"
    "sc5p_v2_hs_PBMC_10k_multi_5gex_5fb_b_t_vdj_t_all_contig_annotations.csv"
)
TENX_LICENSE = "CC BY 4.0"
TARGET_HLAS = {"HLA-A*02:01", "HLA-B*40:01"}
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def download(url: str, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"Using existing {destination.relative_to(ROOT)}")
        return
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "tcr-pmhc-study-data-prep/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded {destination.relative_to(ROOT)} ({destination.stat().st_size:,} bytes)")


def rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {path}")
        yield from reader


def get_column(fieldnames: list[str], requested: str, fallback: tuple[str, ...] = ()) -> str:
    lookup = {name.strip().casefold(): name for name in fieldnames}
    for candidate in (requested, *fallback):
        actual = lookup.get(candidate.casefold())
        if actual:
            return actual
    raise ValueError(f"Missing required CSV column {requested!r}; found {fieldnames!r}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        help="Use the two original IMMREP25 CSVs from this extracted Kaggle directory instead of the public mirror.",
    )
    parser.add_argument("--force", action="store_true", help="Download source files again even if they already exist.")
    args = parser.parse_args()

    if args.kaggle_dir:
        source_root = args.kaggle_dir.expanduser().resolve()
        positive_paths = {
            source: source_root / spec["filename"] for source, spec in IMMREP_FILES.items()
        }
        missing = [path for path in positive_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Kaggle input file(s) not found: " + ", ".join(map(str, missing)))
        positive_origin = "User-provided official Kaggle IMMREP25 competition download"
    else:
        positive_paths = {}
        for source, spec in IMMREP_FILES.items():
            path = IMMREP_ROOT / spec["filename"]
            download(spec["url"], path, args.force)
            positive_paths[source] = path
        positive_origin = "Public mirror of the IMMREP25 competition training CSVs"

    tenx_path = TENX_ROOT / TENX_FILENAME
    download(TENX_URL, tenx_path, args.force)

    positives_path = ROOT / "data" / "positives.tsv"
    positives_path.parent.mkdir(parents=True, exist_ok=True)
    positive_tmp = positives_path.with_suffix(".tsv.tmp")
    positive_rows_by_source: Counter[str] = Counter()
    target_positive_pairs = 0
    target_peptides: set[str] = set()
    known_positive_tcrs: set[str] = set()

    with positive_tmp.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("tcr_beta", "peptide", "hla", "source"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for source, spec in IMMREP_FILES.items():
            path = positive_paths[source]
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError(f"CSV file has no header: {path}")
                peptide_col = get_column(reader.fieldnames, "Peptide", ("epitope",))
                hla_col = get_column(reader.fieldnames, "HLA", ("MHC", "mhc_allele"))
                cdr3_col = get_column(reader.fieldnames, spec["cdr3_column"])
                count = 0
                for row in reader:
                    count += 1
                    peptide = (row.get(peptide_col) or "").strip().upper()
                    hla = (row.get(hla_col) or "").strip().upper().replace("HLA_", "HLA-")
                    tcr_beta = "".join((row.get(cdr3_col) or "").upper().split())
                    writer.writerow({
                        "tcr_beta": tcr_beta,
                        "peptide": peptide,
                        "hla": hla,
                        "source": source,
                    })
                    positive_rows_by_source[source] += 1
                    if hla in TARGET_HLAS and peptide and tcr_beta:
                        target_positive_pairs += 1
                        target_peptides.add(peptide)
                        known_positive_tcrs.add(tcr_beta)
                expected = int(spec["expected_rows"])
                if count != expected:
                    raise ValueError(
                        f"{path.name} contains {count:,} data rows; expected {expected:,}. "
                        "Check that the correct IMMREP25 training file was downloaded."
                    )
    positive_tmp.replace(positives_path)

    repertoire_path = ROOT / "data" / "repertoire.tsv"
    repertoire_tmp = repertoire_path.with_suffix(".tsv.tmp")
    repertoire_sequences: set[str] = set()
    raw_tenx_rows = 0
    selected_tenx_rows = 0
    with tenx_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {tenx_path}")
        required = {name.casefold() for name in ("chain", "productive", "high_confidence", "is_cell", "cdr3")}
        available = {name.strip().casefold() for name in reader.fieldnames}
        missing = required - available
        if missing:
            raise ValueError(f"10x annotation CSV lacks required fields: {sorted(missing)}")
        columns = {name.casefold(): get_column(reader.fieldnames, name) for name in required}
        for row in reader:
            raw_tenx_rows += 1
            if (row.get(columns["chain"]) or "").strip().upper() != "TRB":
                continue
            if (row.get(columns["productive"]) or "").strip().casefold() != "true":
                continue
            if (row.get(columns["high_confidence"]) or "").strip().casefold() != "true":
                continue
            if (row.get(columns["is_cell"]) or "").strip().casefold() != "true":
                continue
            sequence = "".join((row.get(columns["cdr3"]) or "").upper().split())
            if len(sequence) < 3 or not all(residue in CANONICAL_AA for residue in sequence):
                continue
            repertoire_sequences.add(sequence)
            selected_tenx_rows += 1

    if len(repertoire_sequences) < 1000:
        raise ValueError(
            f"Only {len(repertoire_sequences):,} valid unique productive TRB sequences were found; "
            "expected at least 1,000. Check the 10x source file and filters."
        )

    with repertoire_tmp.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("tcr_beta", "hla"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for sequence in sorted(repertoire_sequences):
            writer.writerow({"tcr_beta": sequence, "hla": ""})
    repertoire_tmp.replace(repertoire_path)

    metadata = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "positives": {
            "origin": positive_origin,
            "official_competition": "https://www.kaggle.com/competitions/immrep25",
            "mirror_repository": "https://github.com/Ajaingithub/pMHC_TCR_specificity",
            "raw_files": {
                source: {
                    "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "rows": positive_rows_by_source[source],
                    "sha256": digest(path),
                    "cdr3_beta_column_used": IMMREP_FILES[source]["cdr3_column"],
                }
                for source, path in positive_paths.items()
            },
            "canonical_rows_written": sum(positive_rows_by_source.values()),
            "target_hla_positive_rows_before_farhan_filtering": target_positive_pairs,
            "target_hla_unique_peptides_before_farhan_filtering": len(target_peptides),
            "target_hla_unique_tcr_beta_sequences": len(known_positive_tcrs),
        },
        "repertoire": {
            "origin": "10x Genomics Human PBMC from a Healthy Donor, 10k cells (v2), one donor",
            "url": TENX_URL,
            "dataset_page": "https://www.10xgenomics.com/datasets/human-pbmc-from-a-healthy-donor-10-k-cells-v-2-2-standard-5-0-0",
            "license": TENX_LICENSE,
            "raw_path": str(tenx_path.relative_to(ROOT)),
            "raw_sha256": digest(tenx_path),
            "raw_contig_rows": raw_tenx_rows,
            "rows_after_cell_confidence_productivity_and_trb_filters": selected_tenx_rows,
            "unique_canonical_cdr3_beta_sequences_written": len(repertoire_sequences),
            "known_positive_cdr3_beta_sequences_overlapping_repertoire": len(known_positive_tcrs & repertoire_sequences),
            "filters": ["chain=TRB", "productive=true", "high_confidence=true", "is_cell=true", "canonical amino acids", "unique CDR3 beta"],
        },
        "notes": [
            "The two IMMREP25 training CSVs have the proposal's stated row counts.",
            "The public mirror is used because Kaggle credentials and competition acceptance are not configured on this machine.",
            "The Farhan step filters positives to the two target HLA alleles, deduplicates pairs, and removes known positive TCRs from the repertoire.",
            "The repertoire is from one healthy donor; record this limitation when interpreting results.",
        ],
    }
    metadata_path = ROOT / "data" / "input_sources.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if len(target_peptides) < 5:
        raise ValueError(
            f"Only {len(target_peptides)} target-HLA peptides remain; the configured five-fold split needs at least five."
        )
    print(f"Prepared {positives_path.relative_to(ROOT)}: {sum(positive_rows_by_source.values()):,} raw positives")
    print(f"Prepared {repertoire_path.relative_to(ROOT)}: {len(repertoire_sequences):,} unique healthy-donor TRB sequences")
    print(f"Target-HLA rows available before curation: {target_positive_pairs:,} across {len(target_peptides):,} peptides")
    print("Next command: python3 farhan.py --farhan --config study_config.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
