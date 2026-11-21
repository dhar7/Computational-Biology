#!/usr/bin/env python3
"""Argha's part: shortcut controls, chance thresholds, and uncertainty."""

from __future__ import annotations

from bootstrap_runtime import prefer_project_venv

prefer_project_venv()

import argparse
import math
from pathlib import Path

from study_common import (
    FrozenESM2,
    Progress,
    atomic_json,
    bootstrap_by_peptide,
    cli_part,
    evaluate_scores,
    fit_score_mlp,
    fold_rows,
    get_processed_paths,
    load_config,
    load_processed,
    make_labeled_split,
    metric_values,
    negative_pair_stats,
    permutation_threshold,
    resolve_path,
    set_global_seed,
    summarize_metrics,
    write_tsv,
)


def peptide_metrics(rows, scores, metadata):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["peptide"]), []).append(index)
    output = []
    for peptide, indices in sorted(groups.items()):
        metrics = metric_values([int(rows[i]["label"]) for i in indices], [scores[i] for i in indices])
        if not math.isfinite(metrics["auroc"]):
            continue
        output.append({**metadata, "peptide": peptide, **metrics, "n_pairs": len(indices)})
    return output


def run(config, config_root: Path):
    positives, repertoire, manifest = load_processed(config, config_root)
    output_dir = resolve_path(config_root, config.get("output_dir", "outputs")) / "argha"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment, model_config = config["experiment"], config["model"]
    folds = int(manifest["folds"])
    seeds = [int(seed) for seed in experiment["seeds"]]
    controls = ["full", "peptide_only", "mhc_only", "tcr_blind"]
    jobs = folds * len(seeds) * len(controls)
    progress = Progress("argha", [
        ("Load frozen encoder and processed data", 10),
        ("Train shortcut detector controls", 65),
        ("Label permutation chance thresholds and peptide bootstrap", 20),
        ("Summarize control results", 5),
    ], output_dir, config.get("progress_initial_estimates_minutes", {}).get("argha", 120))
    all_positive_keys = {
        (row["hla"], row["peptide"], row["tcr_beta"]) for row in positives
    }
    encoder = FrozenESM2(model_config)
    progress.tick(0, 1.0, f"loaded {len(positives):,} positives and frozen {model_config['encoder_name']}")

    metrics_rows = []
    peptide_rows = []
    completed = 0
    permutations = int(experiment.get("label_permutations", 1000))
    bootstraps = int(experiment.get("bootstrap_replicates", 1000))
    for fold in range(folds):
        train_positive = fold_rows(positives, fold, train=True)
        test_positive = fold_rows(positives, fold, train=False)
        for seed in seeds:
            run_seed = seed + 100003 * fold
            train_rows = make_labeled_split(
                train_positive, repertoire, all_positive_keys, "repertoire_background", run_seed,
                int(experiment["negatives_per_positive"]), int(experiment.get("hardness_candidate_cap", 5000)),
            )
            test_rows = make_labeled_split(
                test_positive, repertoire, all_positive_keys, "repertoire_background", run_seed + 1,
                int(experiment["negatives_per_positive"]), int(experiment.get("hardness_candidate_cap", 5000)),
            )
            for control in controls:
                set_global_seed(run_seed)
                scores = fit_score_mlp(train_rows, test_rows, encoder, model_config, run_seed, mode=control)
                observed = metric_values([int(row["label"]) for row in test_rows], scores)
                chance = permutation_threshold([int(row["label"]) for row in test_rows], scores, permutations, run_seed + 2)
                bootstrap = bootstrap_by_peptide(test_rows, scores, bootstraps, run_seed + 3)
                result = {
                    "fold": fold,
                    "seed": seed,
                    "control": control,
                    "negative_method": "repertoire_background",
                    "n_train_pairs": len(train_rows),
                    "n_test_pairs": len(test_rows),
                    "positive_prevalence": sum(int(row["label"]) for row in test_rows) / len(test_rows),
                    **negative_pair_stats(test_rows),
                    **observed,
                    **chance,
                    **bootstrap,
                }
                metrics_rows.append(result)
                peptide_rows.extend(peptide_metrics(test_rows, scores, {"fold": fold, "seed": seed, "control": control}))
                completed += 1
                progress.tick(1, completed / jobs, f"fold {fold + 1}/{folds}, seed {seed}, control {control}")
    progress.tick(1, 1.0, f"completed {jobs} control fits")

    write_tsv(output_dir / "shortcut_metrics.tsv", metrics_rows)
    write_tsv(output_dir / "per_peptide_metrics.tsv", peptide_rows)
    summary = summarize_metrics(metrics_rows, ["control"])
    write_tsv(output_dir / "shortcut_summary.tsv", summary)
    seed_spread = summarize_metrics(metrics_rows, ["control", "fold"])
    fold_spread = summarize_metrics(metrics_rows, ["control", "seed"])
    write_tsv(output_dir / "seed_spread_by_fold.tsv", seed_spread)
    write_tsv(output_dir / "fold_spread_by_seed.tsv", fold_spread)
    chance_summary = []
    for control in controls:
        selected = [row for row in metrics_rows if row["control"] == control]
        chance_summary.append({
            "control": control,
            "mean_chance_auroc_p95": sum(row["chance_auroc_p95"] for row in selected) / len(selected),
            "mean_chance_auprc_p95": sum(row["chance_auprc_p95"] for row in selected) / len(selected),
            "permutations_per_fold_seed": permutations,
            "bootstrap_replicates_per_fold_seed": bootstraps,
        })
    write_tsv(output_dir / "chance_thresholds.tsv", chance_summary)
    atomic_json(output_dir / "run_manifest.json", {
        "part": "argha",
        "negative_method": "repertoire_background",
        "controls": controls,
        "folds": folds,
        "seeds": seeds,
        "label_permutations": permutations,
        "bootstrap_replicates": bootstraps,
        "negative_pair_sampling": "Candidate TCRs are sampled without replacement per positive when possible; reused biological pairs have distinct draw-level pair_id values and are counted in each metrics row.",
        "processed_manifest": str(get_processed_paths(config, config_root)["manifest"]),
    })
    progress.tick(2, 1.0, "computed 95th-percentile nulls and peptide-level bootstrap intervals")
    progress.tick(3, 1.0, "wrote shortcut, chance, uncertainty, and per-peptide results")
    progress.tick(4, 1.0, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli_part(parser, "argha")
    args = parser.parse_args()
    if not args.argha:
        parser.error("select this part with --argha")
    config, root = load_config(args.config)
    run(config, root)


if __name__ == "__main__":
    main()
