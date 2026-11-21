# TCR-pMHC negative construction study

This repository implements the proposal in four independently selectable Python parts. Run Farhan's curation once first. After it writes the processed tables and peptide-grouped folds, Argha, Jacob, and Aditi can read those files concurrently; each writes only to its own `outputs/<name>/` directory.

## Environment and inputs

Use Python 3.10 or newer. Create the project-local environment and install the listed packages with:

```text
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The four part scripts automatically re-execute with `.venv` when it exists, so you can then run `python3 farhan.py --farhan --config study_config.json` directly without activating the environment or installing into Homebrew's Python.

The proposal names IMMREP25's IEDB and VDJdb positive training sets (9,652 and 30,274 rows) and a background TCR repertoire. Run the standard-library setup script once to download the two training tables and a public healthy-donor TCRβ repertoire, then convert them to the expected inputs:

```bash
python3 prepare_data.py
```

The script writes `data/positives.tsv`, `data/repertoire.tsv`, and `data/input_sources.json`; raw downloads are kept under `data/raw/`. It uses a public mirror of the two IMMREP25 training CSVs because Kaggle credentials are not required for that mirror. If you prefer the official Kaggle download, accept the competition rules, configure Kaggle's API token, download and extract the competition bundle, then run `python3 prepare_data.py --kaggle-dir data/raw/immrep25-kaggle`. The repertoire is from one healthy PBMC donor, so report that sampling limitation when interpreting the negative sets.

`positives.tsv` is tab-separated and has these columns:

| Column | Meaning |
| --- | --- |
| `tcr_beta` | CDR3β amino-acid sequence |
| `peptide` | Peptide sequence |
| `hla` | HLA allele, for example `HLA-A*02:01` |
| `source` | Optional source label such as `IEDB` or `VDJdb` |

`repertoire.tsv` has a `tcr_beta` column and an empty `hla` column, meaning the repertoire sequences are allele-untyped. Change input column names in `study_config.json` if you substitute tables with another schema. Sequences are uppercased, whitespace is removed, and noncanonical amino acids are filtered. CDR3 anchor residues are retained.

The bundled healthy repertoire is from one donor and is smaller than the negative demand for the most frequent peptide. To keep the proposal's fixed 5:1 row ratio without labeling known binders as negatives, candidates are sampled without replacement across each HLA-peptide group until that group's candidate pool is exhausted, then reused. Each sampled row has a unique `pair_id`; `biological_pair_id` identifies repeated biological pairs. Argha and Aditi metrics report the number of unique negative pairs and reused rows. Reuse is a sampling limitation, not additional independent biological evidence; use a larger independent healthy repertoire for stronger conclusions.

Farhan's manifest checks the raw source row counts against the proposal's 9,652 IEDB and 30,274 VDJdb counts. It separately reports how many records remain after HLA, sequence, and deduplication filters.

## Run a part

Run these commands from the project directory:

```bash
python3 farhan.py --farhan --config study_config.json
python3 argha.py --argha --config study_config.json
python3 jacob.py --jacob --config study_config.json
python3 aditi.py --aditi --config study_config.json
```

Farhan writes canonical positives, a cleaned background repertoire, five peptide-grouped folds, and leakage checks under `data/processed/`. Argha, Jacob, and Aditi require this preprocessing output. Those three commands can then be run in separate terminals or scheduled concurrently. They do not write to one another's result folders. For multiple GPUs, set `model.device` to the appropriate device before launching each process; with one GPU, use CPU inference for parallel Argha/Aditi runs or run them one at a time to avoid device contention.

Every part prints a weighted overall completion percentage, current phase percentage, and estimated time remaining. It also updates `outputs/<part>/progress.json`. The initial ETA in `study_config.json` is a rough first-run estimate; it recalculates from observed throughput as work progresses. Actual time depends on data size, CPU/GPU, and predictor availability.

Initial ETA and overall progress weights are:

| Part | Initial ETA | Progress phases and weights |
| --- | ---: | --- |
| Farhan | 10 min | read/validate 15%, normalize/deduplicate 25%, repertoire/leakage 20%, folds/checks 25%, write outputs 15% |
| Argha | 120 min | load 10%, shortcut fits 65%, permutation/bootstrap 20%, summaries 5% |
| Jacob | 120 min | load 10%, timeboxed predictors 75%, fallback/report 15% |
| Aditi | 240 min | load 8%, main negative curve 60%, robustness 17%, summaries 9%, structure check/figures 6% |

These are startup estimates, not runtime guarantees. During a run, ETA is recalculated from measured progress.

## What each part produces

- **Farhan:** `data/processed/positives.tsv`, `repertoire.tsv`, `folds.tsv`, and `manifest.json`. Identical positive pairs are deduplicated, only the two proposal alleles are retained, known positive TCRs are removed from the background repertoire, and every peptide is assigned to exactly one of five folds.
- **Argha:** shortcut controls using full, peptide-only, MHC-only, and TCR-blind features; AUROC/AUPRC; 1,000 label permutations for the empirical 95th-percentile chance threshold; 1,000 peptide-level bootstrap replicates; per-peptide metrics; and separate seed spread (within fold) and fold spread (within seed) tables.
- **Jacob:** adapter runner for two published predictors, using common folds and negative sets. A command is an argument array in `study_config.json` and can refer to `{train}`, `{test}`, and `{output}`. The test input omits labels. The command must write a TSV with `pair_id` and `score` (or `prediction`) columns. Each predictor has a wall-clock timebox. Optional published-paper fallback records are kept separate from local held-out metrics.
- **Aditi:** the random-pairing and repertoire baselines plus five CDR3β edit-distance strata; the frozen ESM-2 encoder and small MLP; the k-mer/gradient-boosting robustness curve; optional structural plausibility check for up to 200 hardest negatives; per-peptide metrics; separate seed/fold spread tables; and comparison/hardness figures.

All training hyperparameters, seeds, negative ratio, number of folds, uncertainty repetitions, and model checkpoint are in `study_config.json`. The ESM-2 encoder is frozen; only the MLP head is trained. No held-out fold is used to tune hyperparameters.

Hardness is operationalized in code as follows: within an HLA-peptide group, eligible background TCRs are ranked by their minimum CDR3β Levenshtein distance to known binders. The ranked candidates are divided into five equal-count strata. `hard_q99` is the closest stratum and `hard_q1` the most distant. The labels q99, q75, q50, q25, and q1 correspond to these strata; inspect this definition when interpreting the curve.

## Predictor and structure adapters

The proposal does not name the two published predictors, identify exact released model versions, or provide published fallback metrics. `NetTCR-2.2` and `ERGO-II` are example names in the config only. Set their commands to wrappers for the versions your group selected, or change the names. Fallbacks accept either a JSON object containing `auroc`, `auprc`, `reference`, and `context`, or a TSV path with `predictor`, `auroc`, `auprc`, `reference`, and `context` columns. Do not compare literature fallback values as if they were measured on these held-out folds.

The structural plausibility check likewise requires a local wrapper command. It receives `{input}` and `{output}`; input is the sampled candidate TSV and output must contain a `score` or `plausibility` column. Without a configured executable, Aditi records the check as skipped rather than silently treating negatives as validated.

## Static review status

The setup script validates both expected IMMREP25 source row counts and the healthy-donor TRB repertoire schema before writing Farhan's inputs. Full study processing also downloads/caches the ESM-2 checkpoint on first use for Argha/Aditi; Jacob/Aditi integrations require the predictor commands described above.
