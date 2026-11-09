# Selection Biases in LLM Attack Agents — Replication Package

This repository reproduces every number in the paper
*Selection Biases in LLM Attack Agents: Vulnerabilities as Deception Surfaces*.

It contains the full pipeline: the code that runs the experiments (queries the
models and records their choices), the frozen data those runs produced, and the
analysis code that turns that data into the paper's results.

## What the experiments are

Each experiment shows a model a four-option menu of real, published CVEs and asks
which it would attack first. We vary one thing at a time and measure how the
choice shifts.

| Paper | Experiment | Folder |
|-------|-----------|--------|
| E1 | Recognition (CVE identifier hidden vs. shown) | `q1-recognition/` |
| E2 | Prominence vs. severity (CVSS hidden then shown) | `q2-drivers/` |
| E4 | Recency (release year) | `q2-recency/` |
| E3 | Displayed severity | `q3-severity-shown/` |
| E5 | Planted scanner finding | `q4-scan-override/` |
| E6 | Version contradiction | `q5-fabrication/` |
| E7 | Deception warning | `q6-backfire/` |

The folders use a `qN` naming; the `Paper` column above maps each to the
experiment number used in the text. Panel
results for all experiments are in `panel-glmm/`; per-model pick rates with
confidence intervals are in `permodel-summary/`; the prominence index is in
`general/fame/`.

## Setup

Requires Python 3.12+.

```bash
./install.sh                 # creates .venv/ and installs the dependencies
cp .env.example .env         # only needed to re-run the models (not for reproduction)
```

To install the packages manually instead: `pip install -r requirements.txt`.
`OPENROUTER_API_KEY` is only needed to re-run the models; reproduction does not
use it. See `.env.example` for the optional variables.

## Reproduce the paper numbers (fast path, no API key)

The parsed per-trial results are frozen as CSVs in each experiment's `data/`
folder, so you can regenerate every result without re-running any model.

Run everything with one command:

```bash
./reproduce.sh
```

Or run the individual analyses (`reproduce.sh` runs all of these in order):

```bash
python q1-recognition/compute_metrics.py            # E1 recognition-gap (RB) per set/rung
python q2-drivers/compute_e2_pickrates.py           # E2 prominence-vs-severity pick rates
python q3-severity-shown/compute_metrics.py         # E3 displayed-severity pick rates (CSG)
python q2-recency/compute_recency_logodds.py        # E4 recency coefficient
python q4-scan-override/compute_metrics.py          # E5 planted-finding redirection per menu
python q4-scan-override/ladder/ladder_glmm.py       # E5 finding-cue decomposition ladder
python q5-fabrication/abstain/abstain_analysis.py   # E6 version-contradiction abstain analysis
python q6-backfire/caution/caution_analysis.py      # E7 deception-warning caution-placebo
python panel-glmm/panel_glmm.py                     # panel coefficients across experiments
python permodel-summary/wilson_appendix.py          # per-model pick-rate Wilson table
python general/fame/fame_rank.py                    # prominence index
python validity/recognition_robustness.py           # robustness: recognition gap (Table 8)
python validity/scanflag_robustness.py              # robustness: scan-flag pull (Table 9)
```

Each script prints its results; compare them against the paper's reported values.

## Re-run the experiments from scratch (full path, needs an API key)

```bash
python engine/run_experiment.py --config configs/<config>.yaml
```

`engine/run_experiment.py` queries the seven models through OpenRouter,
`engine/matching.py` parses each response into a recorded choice, and the run is
written to both a full JSONL capture and a flat CSV. Re-running costs money in
API credits and will not reproduce the exact trials (models are sampled), but it
should reproduce the same measured effects.

## Data: CSV vs. JSONL

- **`<experiment>/data/*.csv`** — the parsed results. One row per trial with the
  fields the analysis needs (model, menu, information level, the candidate the
  model picked, whether the pick was valid). The analysis scripts read these.
- **`raw-responses/*.jsonl`** — the complete raw capture. One record per trial
  with the full prompt, the model's entire response, reasoning tokens, and
  token counts. The CSVs are derived from these. The raw JSONL is large.

Every paper number reproduces from the CSVs alone. The JSONL is only needed to
re-parse from the raw model outputs or to read full model rationales.

## Layout

```
engine/            run the experiments (models -> JSONL + CSV)
configs/           one YAML per experiment run
pool/              the CVE candidate set
recon/             simulated scanner inputs
q1-recognition/    E1 analysis + data
q2-drivers/        E2 analysis + data
q2-recency/        E4 analysis + data
q3-severity-shown/ E3 analysis + data
q4-scan-override/  E5 analysis + data
q5-fabrication/    E6 analysis + data
q6-backfire/       E7 analysis + data
panel-glmm/        panel coefficients across experiments
permodel-summary/  per-model Wilson table
general/fame/      prominence index (fame_rank)
raw-responses/     raw per-trial JSONL (one record per trial; large)
```
