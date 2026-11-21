#!/usr/bin/env python3
"""Jacob's part: timeboxed published-predictor adapters and fallbacks.

Each configured predictor command receives {train}, {test}, and {output} paths.
The test table omits labels. The predictor must write a TSV with pair_id and
score columns. Published-number fallbacks are kept separate from local metrics.
"""

from __future__ import annotations

from bootstrap_runtime import prefer_project_venv

prefer_project_venv()

import argparse
import math
import time
from pathlib import Path

from study_common import (
    Progress,
    atomic_json,
    cli_part,
    fold_rows,
    get_processed_paths,
    load_config,
    load_processed,
    make_labeled_split,
    metric_values,
    negative_pair_stats,
    read_tsv,
    resolve_path,
    run_external_predictor,
    write_tsv,
)


def clean_model_rows(rows, include_labels: bool):
    return [
        {
            "pair_id": row["pair_id"],
            "tcr_beta": row["tcr_beta"],
            "peptide": row["peptide"],
            "hla": row["hla"],
            **({"label": row["label"]} if include_labels and "label" in row else {}),
        }
        for row in rows
    ]


def read_fallback(fallback, predictor_name: str, config_root: Path):
    if fallback is None:
        return None
    if isinstance(fallback, dict):
        return dict(fallback)
    path = resolve_path(config_root, str(fallback))
    rows = read_tsv(path)
    for row in rows:
        if row.get("predictor") == predictor_name:
            result = dict(row)
            for key in ("auroc", "auprc"):
                try:
                    result[key] = float(result[key])
                except (KeyError, ValueError):
                    result[key] = float("nan")
            return result
    return None


def load_prediction_scores(path: Path, test_rows):
    predictions = read_tsv(path)
    if not predictions:
        raise ValueError("predictor output has no prediction rows")
    if "pair_id" not in predictions[0]:
        raise ValueError("predictor output must contain pair_id")
    score_col = "score" if "score" in predictions[0] else ("prediction" if "prediction" in predictions[0] else None)
    if score_col is None:
        raise ValueError("predictor output must contain score or prediction")
    score_by_id = {}
    for row in predictions:
        pair_id = row["pair_id"]
        if pair_id in score_by_id:
            raise ValueError(f"duplicate prediction for pair_id {pair_id}")
        value = float(row[score_col])
        if not math.isfinite(value):
            raise ValueError(f"non-finite prediction for pair_id {pair_id}")
        score_by_id[pair_id] = value
    expected = {row["pair_id"] for row in test_rows}
    if set(score_by_id) != expected:
        missing = len(expected - set(score_by_id))
        extra = len(set(score_by_id) - expected)
        raise ValueError(f"prediction pair IDs do not match test set (missing={missing}, extra={extra})")
    return [score_by_id[row["pair_id"]] for row in test_rows]


def run(config, config_root: Path):
    positives, repertoire, processed_manifest = load_processed(config, config_root)
    output_dir = resolve_path(config_root, config.get("output_dir", "outputs")) / "jacob"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    folds = int(processed_manifest["folds"])
    seeds = [int(seed) for seed in experiment["seeds"]]
    methods = ["random_pairing", "repertoire_background"] + [f"hard_q{int(v)}" for v in experiment.get("hardness_percentiles", [99, 75, 50, 25, 1])]
    predictors = config.get("predictors", [])
    all_positive_keys = {(row["hla"], row["peptide"], row["tcr_beta"]) for row in positives}
    input_dir = output_dir / "predictor_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress("jacob", [
        ("Load processed data and predictor adapters", 10),
        ("Run timeboxed predictors on common folds and negatives", 75),
        ("Apply published-number fallbacks and summarize", 15),
    ], output_dir, config.get("progress_initial_estimates_minutes", {}).get("jacob", 60))
    progress.tick(0, 1.0, f"loaded {len(positives):,} positive pairs and {len(predictors)} predictor configurations")

    result_rows = []
    status_rows = []
    expected_jobs = len(predictors) * folds * len(seeds) * len(methods)
    completed = 0
    for predictor in predictors:
        name = str(predictor.get("name", "unnamed_predictor"))
        command = predictor.get("command", [])
        timeout_seconds = int(predictor.get("timeout_seconds", 3600))
        deadline = time.monotonic() + timeout_seconds
        fallback = read_fallback(predictor.get("published_fallback"), name, config_root)
        if fallback is not None:
            result_rows.append({
                "predictor": name,
                "fold": "",
                "seed": "",
                "method": "published_fallback",
                "status": "published_context_not_comparable",
                "auroc": fallback.get("auroc", ""),
                "auprc": fallback.get("auprc", ""),
                "reference": fallback.get("reference", ""),
                "context": fallback.get("context", ""),
            })
            status_rows.append({"predictor": name, "status": "published_fallback_added", "details": "Kept separate from held-out local metrics."})
        if not command:
            status_rows.append({
                "predictor": name,
                "status": "not_configured",
                "details": "Add a command adapter to study_config.json or provide published fallback numbers.",
            })
            completed += folds * len(seeds) * len(methods)
            progress.tick(1, completed / max(1, expected_jobs), f"{name}: command not configured")
            continue
        for fold in range(folds):
            train_positive = fold_rows(positives, fold, train=True)
            test_positive = fold_rows(positives, fold, train=False)
            for seed in seeds:
                run_seed = seed + 200003 * fold
                train_hard_cache = {}
                test_hard_cache = {}
                for method in methods:
                    train_rows = make_labeled_split(
                        train_positive, repertoire, all_positive_keys, method, run_seed,
                        int(experiment["negatives_per_positive"]), int(experiment.get("hardness_candidate_cap", 5000)), train_hard_cache,
                    )
                    test_rows = make_labeled_split(
                        test_positive, repertoire, all_positive_keys, method, run_seed + 1,
                        int(experiment["negatives_per_positive"]), int(experiment.get("hardness_candidate_cap", 5000)), test_hard_cache,
                    )
                    job_dir = input_dir / name.replace("/", "_") / method / f"fold_{fold}_seed_{seed}"
                    job_dir.mkdir(parents=True, exist_ok=True)
                    train_path = job_dir / "train.tsv"
                    test_path = job_dir / "test.tsv"
                    prediction_path = job_dir / "predictions.tsv"
                    write_tsv(train_path, clean_model_rows(train_rows, include_labels=True))
                    write_tsv(test_path, clean_model_rows(test_rows, include_labels=False))
                    if prediction_path.exists():
                        prediction_path.unlink()
                    remaining = max(0, int(deadline - time.monotonic()))
                    okay = False
                    detail = "timebox expired"
                    if remaining > 0:
                        okay, detail = run_external_predictor(
                            command, test_path, prediction_path, remaining,
                            placeholders={"train": str(train_path), "test": str(test_path), "output": str(prediction_path)},
                        )
                    row_base = {
                        "predictor": name,
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "hardness_percentile": method.removeprefix("hard_q") if method.startswith("hard_q") else "",
                    }
                    if okay:
                        try:
                            scores = load_prediction_scores(prediction_path, test_rows)
                            metrics = metric_values([int(row["label"]) for row in test_rows], scores)
                            result_rows.append({
                                **row_base,
                                "status": "evaluated",
                                **metrics,
                                "n_test_pairs": len(test_rows),
                                **negative_pair_stats(test_rows),
                            })
                            status_rows.append({**row_base, "status": "evaluated", "details": detail})
                        except (ValueError, OSError) as exc:
                            status_rows.append({**row_base, "status": "invalid_output", "details": str(exc)})
                    else:
                        status_rows.append({**row_base, "status": "unavailable", "details": detail})
                    completed += 1
                    progress.tick(1, completed / max(1, expected_jobs), f"{name}, fold {fold + 1}/{folds}, seed {seed}, {method}: {status_rows[-1]['status']}")
    progress.tick(1, 1.0, f"processed {completed} predictor jobs")

    write_tsv(output_dir / "predictor_metrics.tsv", result_rows, fieldnames=[
        "predictor", "fold", "seed", "method", "hardness_percentile", "status",
        "auroc", "auprc", "n_test_pairs", "reference", "context",
        "n_negative_rows", "n_unique_negative_pairs", "reused_negative_rows", "negative_pair_reuse_fraction",
    ])
    write_tsv(output_dir / "predictor_status.tsv", status_rows, fieldnames=[
        "predictor", "fold", "seed", "method", "status", "details",
    ])
    atomic_json(output_dir / "run_manifest.json", {
        "part": "jacob",
        "predictors": [predictor.get("name", "unnamed_predictor") for predictor in predictors],
        "folds": folds,
        "seeds": seeds,
        "negative_methods": methods,
        "timebox_seconds_per_predictor": {predictor.get("name", "unnamed_predictor"): int(predictor.get("timeout_seconds", 3600)) for predictor in predictors},
        "test_inputs_omit_labels": True,
        "processed_manifest": str(get_processed_paths(config, config_root)["manifest"]),
        "notes": [
            "The proposal does not name the two published predictors; NetTCR-2.2 and ERGO-II are configurable placeholders in study_config.json.",
            "Published-paper metrics are reported separately and are not treated as measurements on this study's held-out folds.",
            "Configure predictor commands as argument arrays using {train}, {test}, and {output}; predictor output must be a TSV with pair_id and score.",
            "Negative draw rows have unique pair_id values; biological_pair_id in local analysis rows identifies any reused pair when the candidate pool is exhausted.",
        ],
    })
    progress.tick(2, 1.0, "wrote local predictor results, timebox statuses, and separate fallback records")
    progress.tick(3, 1.0, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_part(parser, "jacob")
    args = parser.parse_args()
    if not args.jacob:
        parser.error("select this part with --jacob")
    config, root = load_config(args.config)
    run(config, root)


if __name__ == "__main__":
    main()
