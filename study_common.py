"""Shared, deterministic helpers for the TCR-pMHC negative-set study.

Input tables are tab-separated by default.  The canonical columns written by
Farhan's preprocessing step are: tcr_beta, peptide, hla, source, fold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_ALLELES = frozenset(("HLA-A*02:01", "HLA-B*40:01"))
DEFAULT_METHODS = ("random_pairing", "repertoire_background")


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")
    return config, config_path.parent


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    iterator = iter(rows)
    first = next(iterator, None)
    if fieldnames is None:
        fieldnames = list(first.keys()) if first is not None else []
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(iterator)
    temp.replace(path)


def read_tsv(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Input file has no header: {path}")
        return [dict(row) for row in reader]


def normalize_aa(value: Any) -> str:
    if value is None:
        return ""
    return "".join(str(value).upper().split())


def is_valid_aa(sequence: str, min_length: int = 1) -> bool:
    return len(sequence) >= min_length and all(residue in AA for residue in sequence)


def normalize_hla(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper().replace("HLA_", "HLA-")
    match = re.fullmatch(r"(?:HLA-)?([AB])\*?(\d{2}):?(\d{2})", text)
    if not match:
        return ""
    locus, first, second = match.groups()
    return f"HLA-{locus}*{first}:{second}"


def stable_pair_id(hla: str, peptide: str, tcr_beta: str) -> str:
    payload = "\x1f".join((hla, peptide, tcr_beta)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def stable_draw_id(
    biological_pair_id: str, method: str, seed: int, positive_pair_id: str, draw_index: int,
) -> str:
    """Return a unique row ID even when a biological negative is sampled again."""
    payload = "\x1f".join((biological_pair_id, method, str(seed), positive_pair_id, str(draw_index)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class Progress:
    """Weighted task progress with live ETA and a machine-readable status file."""

    def __init__(self, part: str, stages: Sequence[tuple[str, float]], output: Path, initial_minutes: float):
        if not stages or any(weight <= 0 for _, weight in stages):
            raise ValueError("Progress stages must have positive weights")
        total = sum(weight for _, weight in stages)
        self.stages = [(name, weight / total) for name, weight in stages]
        self.part = part
        self.output = output / "progress.json"
        self.started = time.monotonic()
        self.initial_seconds = max(1.0, float(initial_minutes) * 60.0)
        self.index = -1
        self.stage_fraction = 0.0
        self.ema_seconds_per_fraction: float | None = None
        self.last_tick = self.started
        self.last_total_fraction = 0.0
        self.current_stage = "Starting"
        self.tick(-1, 0.0, force=True)

    def _overall(self, index: int, fraction: float) -> float:
        return sum(weight for _, weight in self.stages[:max(index, 0)]) + (
            self.stages[index][1] * fraction if 0 <= index < len(self.stages) else 0.0
        )

    def tick(self, index: int, fraction: float, detail: str = "", force: bool = False) -> None:
        now = time.monotonic()
        fraction = min(1.0, max(0.0, float(fraction)))
        if index >= len(self.stages):
            index = len(self.stages) - 1
            fraction = 1.0
        overall = 0.0 if index < 0 else self._overall(index, fraction)
        elapsed = now - self.started
        delta_fraction = overall - self.last_total_fraction
        delta_time = now - self.last_tick
        if delta_fraction > 0 and delta_time > 0:
            sample_rate = delta_time / delta_fraction
            self.ema_seconds_per_fraction = (
                sample_rate if self.ema_seconds_per_fraction is None
                else 0.35 * sample_rate + 0.65 * self.ema_seconds_per_fraction
            )
        if index < 0:
            stage_name = "Starting"
        else:
            stage_name = self.stages[index][0]
        if overall >= 1.0:
            eta = 0.0
        elif self.ema_seconds_per_fraction is not None:
            eta = self.ema_seconds_per_fraction * (1.0 - overall)
        else:
            eta = self.initial_seconds * (1.0 - overall)
        payload = {
            "part": self.part,
            "progress_percent": round(100.0 * overall, 2),
            "stage": stage_name,
            "stage_percent": round(100.0 * fraction, 2),
            "detail": detail,
            "elapsed_seconds": round(elapsed, 1),
            "estimated_seconds_remaining": round(eta, 1),
            "updated_unix": time.time(),
        }
        self.current_stage = stage_name
        if force or delta_fraction > 0 or detail:
            print(
                f"[{self.part}] {payload['progress_percent']:.1f}% overall | "
                f"{stage_name} {payload['stage_percent']:.1f}% | "
                f"ETA {format_duration(eta)}" + (f" | {detail}" if detail else ""),
                flush=True,
            )
            atomic_json(self.output, payload)
        self.last_tick = now
        self.last_total_fraction = overall


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_processed_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    processed = resolve_path(root, config["data"]["processed_dir"])
    return {
        "dir": processed,
        "positives": processed / "positives.tsv",
        "repertoire": processed / "repertoire.tsv",
        "manifest": processed / "manifest.json",
    }


def load_processed(config: Mapping[str, Any], root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    paths = get_processed_paths(config, root)
    if not paths["manifest"].exists():
        raise FileNotFoundError(f"Processed dataset missing. Run farhan.py --farhan first. Expected {paths['manifest']}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    positives = read_tsv(paths["positives"])
    repertoire = read_tsv(paths["repertoire"])
    if not positives or not repertoire:
        raise ValueError("Processed positives and repertoire must both contain at least one row")
    if any(not row.get("fold", "").isdigit() for row in positives):
        raise ValueError("Positive table lacks integer fold assignments; rerun Farhan preprocessing")
    return positives, repertoire, manifest


def positive_pair_keys(rows: Iterable[Mapping[str, str]]) -> set[tuple[str, str, str]]:
    return {(row["hla"], row["peptide"], row["tcr_beta"]) for row in rows}


def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return int(Levenshtein.distance(a, b))
    except ImportError:
        if len(a) < len(b):
            a, b = b, a
        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, 1):
            current = [i]
            for j, char_b in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
            previous = current
        return previous[-1]


def make_negative_pairs(
    positives: Sequence[Mapping[str, str]],
    repertoire: Sequence[Mapping[str, str]],
    all_positive_keys: set[tuple[str, str, str]],
    method: str,
    seed: int,
    negatives_per_positive: int,
    hardness_cap: int,
    hardness_cache: dict[tuple[str, str], list[tuple[str, int]]] | None = None,
) -> list[dict[str, Any]]:
    """Sample deterministic negative rows for one fold split.

    The proposal fixes the negative-to-positive row ratio at five-to-one.
    Candidate sequences are sampled without replacement within each
    HLA-peptide group until its candidate pool is exhausted. Further draws
    reuse biological pairs with distinct pair_id values and retain
    biological_pair_id so reuse is auditable. Known binders for the same
    HLA-peptide pair are always excluded. Hardness strata partition ranked
    edit distances into five bins; q99 is closest and q1 most distant.
    """
    if method not in DEFAULT_METHODS and not re.fullmatch(r"hard_q(?:99|75|50|25|1)", method):
        raise ValueError(f"Unknown negative construction method: {method}")
    if int(negatives_per_positive) < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    rng = random.Random(int(seed))
    known_tcrs = {tcr for _, _, tcr in all_positive_keys}
    if method == "random_pairing":
        pool_by_hla = {
            hla: sorted({tcr for positive_hla, _, tcr in all_positive_keys if positive_hla == hla})
            for hla in {row["hla"] for row in positives}
        }
    else:
        pool_by_hla: dict[str, list[str]] = {}
        target_hlas = {row["hla"] for row in positives}
        for target_hla in target_hlas:
            pool_by_hla[target_hla] = sorted({
                row["tcr_beta"] for row in repertoire
                if row["tcr_beta"] not in known_tcrs and row.get("hla", "") in ("", target_hla)
            })

    def candidates_for_hla(hla: str) -> list[str]:
        candidates = pool_by_hla.get(hla, [])
        if not candidates:
            raise ValueError(f"No eligible TCRs for HLA {hla} and method {method}; inspect repertoire coverage")
        return candidates

    binders_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    known_binders_by_peptide: dict[str, set[str]] = defaultdict(set)
    for hla, peptide, tcr in all_positive_keys:
        binders_by_pair[(hla, peptide)].append(tcr)
        known_binders_by_peptide[peptide].add(tcr)

    hard_candidates = hardness_cache if hardness_cache is not None else {}
    if method.startswith("hard_q"):
        if int(hardness_cap) < 1:
            raise ValueError("hardness_candidate_cap must be at least 1 for hard-negative methods")
        for row in positives:
            group_key = (row["hla"], row["peptide"])
            if group_key in hard_candidates:
                continue
            binders = binders_by_pair[group_key]
            eligible = [tcr for tcr in candidates_for_hla(group_key[0]) if tcr not in known_binders_by_peptide[group_key[1]]]
            if not eligible:
                raise ValueError(f"No eligible hard-negative candidates for {group_key}")
            if len(eligible) > hardness_cap:
                eligible = sorted(rng.sample(eligible, hardness_cap))
            ranked = [(tcr, min(levenshtein(tcr, binder) for binder in binders)) for tcr in eligible]
            ranked.sort(key=lambda item: (item[1], item[0]))
            hard_candidates[group_key] = ranked

    try:
        q_value = int(method.removeprefix("hard_q")) if method.startswith("hard_q") else None
    except ValueError:
        q_value = None

    positives_by_pair: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for positive in positives:
        positives_by_pair[(positive["hla"], positive["peptide"])].append(positive)

    output: list[dict[str, Any]] = []
    for hla, peptide in sorted(positives_by_pair):
        pair_positives = positives_by_pair[(hla, peptide)]
        if method.startswith("hard_q"):
            ranked = hard_candidates[(hla, peptide)]
            bin_number = {99: 0, 75: 1, 50: 2, 25: 3, 1: 4}[q_value]
            left = math.floor(bin_number * len(ranked) / 5)
            right = math.floor((bin_number + 1) * len(ranked) / 5)
            candidates = [item[0] for item in ranked[left:right]] or [ranked[min(left, len(ranked) - 1)][0]]
            distance_by_tcr = dict(ranked)
        else:
            candidates = candidates_for_hla(hla)
            distance_by_tcr = {}
        eligible = [tcr for tcr in candidates if tcr not in known_binders_by_peptide[peptide]]
        if not eligible:
            raise ValueError(f"No eligible negative candidates remain for {hla}, {peptide}, method={method}")
        count = int(negatives_per_positive)
        positive_order = list(pair_positives)
        rng.shuffle(positive_order)
        candidate_order = list(eligible)
        rng.shuffle(candidate_order)
        for positive_index, positive in enumerate(positive_order):
            start = positive_index * count
            selected = [candidate_order[(start + draw_index) % len(candidate_order)] for draw_index in range(count)]
            positive_id = stable_pair_id(hla, peptide, positive["tcr_beta"])
            for draw_index, tcr in enumerate(selected):
                biological_id = stable_pair_id(hla, peptide, tcr)
                output.append({
                    "pair_id": stable_draw_id(biological_id, method, seed, positive_id, draw_index),
                    "biological_pair_id": biological_id,
                    "tcr_beta": tcr,
                    "peptide": peptide,
                    "hla": hla,
                    "label": 0,
                    "method": method,
                    "hardness_percentile": q_value if q_value is not None else "",
                    "nearest_edit_distance": distance_by_tcr.get(tcr, ""),
                    "positive_pair_id": positive_id,
                })
    return output


def make_labeled_split(
    positives: Sequence[Mapping[str, str]],
    repertoire: Sequence[Mapping[str, str]],
    all_positive_keys: set[tuple[str, str, str]],
    method: str,
    seed: int,
    negatives_per_positive: int,
    hardness_cap: int,
    hardness_cache: dict[tuple[str, str], list[tuple[str, int]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in positives:
        pair_id = stable_pair_id(row["hla"], row["peptide"], row["tcr_beta"])
        rows.append({
            "pair_id": pair_id,
            "biological_pair_id": pair_id,
            "tcr_beta": row["tcr_beta"],
            "peptide": row["peptide"],
            "hla": row["hla"],
            "label": 1,
            "method": method,
            "hardness_percentile": "",
            "nearest_edit_distance": "",
            "positive_pair_id": stable_pair_id(row["hla"], row["peptide"], row["tcr_beta"]),
        })
    rows.extend(make_negative_pairs(
        positives, repertoire, all_positive_keys, method, seed,
        negatives_per_positive, hardness_cap, hardness_cache,
    ))
    return rows


def negative_pair_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Count negative draw rows and distinct biological negative pairs."""
    negatives = [row for row in rows if int(row["label"]) == 0]
    biological_ids = {
        str(row.get("biological_pair_id") or row["pair_id"])
        for row in negatives
    }
    count = len(negatives)
    unique = len(biological_ids)
    return {
        "n_negative_rows": count,
        "n_unique_negative_pairs": unique,
        "reused_negative_rows": count - unique,
        "negative_pair_reuse_fraction": (count - unique) / count if count else 0.0,
    }


def metric_values(y_true: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(y_score, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def permutation_threshold(y_true: Sequence[int], y_score: Sequence[float], permutations: int, seed: int) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(int(permutations)):
        shuffled = rng.permutation(y)
        metrics = metric_values(shuffled, scores)
        if math.isfinite(metrics["auroc"]):
            aurocs.append(metrics["auroc"])
            auprcs.append(metrics["auprc"])
    return {
        "chance_auroc_p95": float(np.quantile(aurocs, 0.95)) if aurocs else float("nan"),
        "chance_auprc_p95": float(np.quantile(auprcs, 0.95)) if auprcs else float("nan"),
    }


def bootstrap_by_peptide(
    rows: Sequence[Mapping[str, Any]], y_score: Sequence[float], replicates: int, seed: int,
) -> dict[str, float]:
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    peptides = sorted({str(row["peptide"]) for row in rows})
    indices_by_peptide: dict[str, np.ndarray] = {
        peptide: np.asarray([i for i, row in enumerate(rows) if str(row["peptide"]) == peptide], dtype=np.int64)
        for peptide in peptides
    }
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(int(replicates)):
        sampled = rng.choice(peptides, size=len(peptides), replace=True)
        indices = np.concatenate([indices_by_peptide[str(peptide)] for peptide in sampled])
        metrics = metric_values(y[indices], scores[indices])
        if math.isfinite(metrics["auroc"]):
            aurocs.append(metrics["auroc"])
            auprcs.append(metrics["auprc"])
    return {
        "bootstrap_auroc_low": float(np.quantile(aurocs, 0.025)) if aurocs else float("nan"),
        "bootstrap_auroc_high": float(np.quantile(aurocs, 0.975)) if aurocs else float("nan"),
        "bootstrap_auprc_low": float(np.quantile(auprcs, 0.025)) if auprcs else float("nan"),
        "bootstrap_auprc_high": float(np.quantile(auprcs, 0.975)) if auprcs else float("nan"),
    }


class FrozenESM2:
    """Frozen ESM-2 mean-pooling encoder with a per-process sequence cache."""

    def __init__(self, model_config: Mapping[str, Any]):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Model runs require torch and transformers; install requirements.txt") from exc
        self.torch = torch
        requested_device = str(model_config.get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Configured device {requested_device!r} is unavailable")
        self.device = torch.device(requested_device)
        name = str(model_config["encoder_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.batch_size = int(model_config.get("batch_size", 32))
        self.max_length = int(model_config.get("max_length", 512))
        self.dimension = int(self.model.config.hidden_size)
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        unique = list(dict.fromkeys(sequences))
        missing = [sequence for sequence in unique if sequence not in self.cache]
        torch = self.torch
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            tokens = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length, add_special_tokens=True,
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
            attention = tokens["attention_mask"]
            for index, sequence in enumerate(batch):
                length = int(attention[index].sum().item())
                if length <= 2:
                    raise ValueError(f"Sequence was truncated to no amino-acid tokens: {sequence!r}")
                vector = hidden[index, 1:length - 1].mean(dim=0).float().cpu().numpy()
                self.cache[sequence] = vector
        return np.stack([self.cache[sequence] for sequence in sequences]).astype(np.float32, copy=False)


def pair_features(rows: Sequence[Mapping[str, Any]], encoder: FrozenESM2, mode: str) -> np.ndarray:
    peptides = [str(row["peptide"]) for row in rows]
    tcrs = [str(row["tcr_beta"]) for row in rows]
    allele_idx = np.asarray([0 if row["hla"] == "HLA-A*02:01" else 1 for row in rows], dtype=np.float32)
    if mode == "mhc_only":
        allele = np.zeros((len(rows), 2), dtype=np.float32)
        allele[np.arange(len(rows)), allele_idx.astype(np.int64)] = 1.0
        return allele
    peptide_embeddings = encoder.encode(peptides)
    if mode == "peptide_only":
        return peptide_embeddings
    allele = np.zeros((len(rows), 2), dtype=np.float32)
    allele[np.arange(len(rows)), allele_idx.astype(np.int64)] = 1.0
    if mode == "tcr_blind":
        return np.concatenate((peptide_embeddings, allele), axis=1)
    if mode != "full":
        raise ValueError(f"Unknown feature mode: {mode}")
    tcr_embeddings = encoder.encode(tcrs)
    return np.concatenate((tcr_embeddings, peptide_embeddings, allele), axis=1)


def fit_score_mlp(
    train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]],
    encoder: FrozenESM2, model_config: Mapping[str, Any], seed: int, mode: str = "full",
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(seed)
    x_train = pair_features(train_rows, encoder, mode)
    x_test = pair_features(test_rows, encoder, mode)
    y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.float32)
    device = encoder.device
    hidden_dim = int(model_config.get("hidden_dim", 128))
    dropout = float(model_config.get("dropout", 0.2))
    net = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=float(model_config.get("learning_rate", 1e-3)),
        weight_decay=float(model_config.get("weight_decay", 1e-4)),
    )
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        dataset, batch_size=int(model_config.get("train_batch_size", 256)),
        shuffle=True, num_workers=int(model_config.get("num_workers", 0)),
    )
    net.train()
    for _ in range(int(model_config.get("epochs", 10))):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(batch_x).squeeze(1), batch_y)
            loss.backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(x_test).to(device)).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def kmer_rows(rows: Sequence[Mapping[str, Any]], model_config: Mapping[str, Any]) -> np.ndarray:
    from sklearn.feature_extraction import FeatureHasher
    feature_count = int(model_config.get("gradient_boosting_features", 256))
    hasher = FeatureHasher(n_features=feature_count, input_type="dict", alternate_sign=False, dtype=np.float32)
    records: list[dict[str, float]] = []
    for row in rows:
        features: dict[str, float] = defaultdict(float)
        for prefix, sequence in (("tcr", str(row["tcr_beta"])), ("pep", str(row["peptide"]))):
            for i in range(max(0, len(sequence) - 2)):
                features[f"{prefix}:{sequence[i:i + 3]}"] += 1.0
        features[f"hla:{row['hla']}"] += 1.0
        records.append(features)
    return hasher.transform(records).toarray().astype(np.float32, copy=False)


def fit_score_gradient_boosting(
    train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any], seed: int,
) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    x_train = kmer_rows(train_rows, model_config)
    x_test = kmer_rows(test_rows, model_config)
    y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    clf = HistGradientBoostingClassifier(
        max_iter=int(model_config.get("gradient_boosting_iterations", 100)),
        max_leaf_nodes=15, learning_rate=0.1, random_state=seed,
    )
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def summarize_metrics(rows: Sequence[Mapping[str, Any]], group_columns: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(column, "")) for column in group_columns)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(group_columns, key))
        for metric in ("auroc", "auprc"):
            values = np.asarray([float(row[metric]) for row in group if _is_finite(row.get(metric))], dtype=float)
            result[f"mean_{metric}"] = float(np.mean(values)) if len(values) else float("nan")
            result[f"sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            result[f"n_{metric}"] = int(len(values))
        output.append(result)
    return output


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_external_predictor(
    command_template: Sequence[str], input_path: Path | None, output_path: Path,
    timeout_seconds: int, placeholders: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    if not command_template:
        return False, "predictor command not configured"
    replacements = {"input": str(input_path or ""), "output": str(output_path)}
    replacements.update({key: str(value) for key, value in (placeholders or {}).items()})
    command = [str(token).format(**replacements) for token in command_template]
    executable = shutil.which(command[0])
    if executable is None and not Path(command[0]).exists():
        return False, f"predictor executable not found: {command[0]}"
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=int(timeout_seconds), check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_seconds} seconds"
    except OSError as exc:
        return False, f"could not launch predictor: {exc}"
    if completed.returncode != 0:
        message = completed.stderr.strip()[-2000:] or f"exit status {completed.returncode}"
        return False, message
    if not output_path.exists():
        return False, "predictor exited successfully but produced no output file"
    return True, completed.stdout.strip()[-2000:]


def evaluate_scores(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float],
    permutations: int, bootstraps: int, seed: int,
) -> dict[str, float]:
    y = [int(row["label"]) for row in rows]
    result = metric_values(y, scores)
    result.update(permutation_threshold(y, scores, permutations, seed))
    result.update(bootstrap_by_peptide(rows, scores, bootstraps, seed + 1))
    return result


def fold_rows(positives: Sequence[Mapping[str, str]], fold: int, train: bool) -> list[dict[str, str]]:
    target = [row for row in positives if (int(row["fold"]) != fold if train else int(row["fold"]) == fold)]
    return target


def cli_part(parser: Any, part: str) -> None:
    parser.add_argument(f"--{part}", action="store_true", help=f"run the {part.title()} part")
    parser.add_argument("--config", default="study_config.json", help="path to study_config.json")


def require_part_flag(args: Any, part: str) -> None:
    if not getattr(args, part):
        raise SystemExit(f"Select this part explicitly: python3 {part}.py --{part} --config study_config.json")

