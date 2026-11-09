# honeyquest-for-llms

Self-contained copy of the Honeyquest agent-benchmark harness. Runs on any
machine with no other repo cloned. Everything the run needs (code + query sets
+ ground truth) is in this folder.

## What's here

- `src/` — the experiment code (entry point `run.py`, the query loader, the
  prompt, and one runner per provider).
- `data/` — the baked query sets and ground truth the run reads:
  - `queries_canonical_30.json` + `ground_truth.json` — the 30 the human cohort sees.
  - `queries_full_174.json` + `ground_truth_full_174.json` — the full set.
- `results/` — output lands here as JSONL + a sidecar manifest.

- `analysis/` — scoring code:
  - `metrics.py` — self-contained CLI. Computes the core per-model and cohort
    metrics from your `results/*.jsonl` against the ground truth. Stdlib only.
  - `paper_stats.py` — the B1 (fell-for-trap binomial) and B2 (McNemar
    attention-diversion) stat functions plus the paper-table renderers. This is
    a function library, NOT a script. It needs `pandas` + `scipy`, reads
    `data/honeyquest_queries.csv` for the B2 query pairings, and is only driven
    by `build_explainer.py` (not included here). On its own it computes nothing.

- `paper_results/` — the final LLM runs behind the paper: 21 model files, one
  per model, 10,962 responses total (21 models x 3 trials x 174 queries). Score
  these to reproduce the paper's numbers.
- `honeyquest_dataset/` — the upstream Honeyquest dataset (Dynatrace, Apache 2.0):
  the raw query YAMLs and the 47-participant human baseline (`human_marks.csv`).
  See that folder's README for source and license.

Figure scripts are left out. This folder runs the experiment and the core
metrics; full paper-table reproduction needs the rest of the upstream
`analysis/` tree.

## Score a run

```sh
PYTHONPATH=src python analysis/metrics.py results/*.jsonl
```

## Reproduce the paper numbers

```sh
PYTHONPATH=src python analysis/metrics.py paper_results/*.jsonl --group-by none
```

Cohort aggregate: fell-for-trap 78.5% (3513/4473), exploit-on-neutral 93%,
risk-detected 78%. B1 (first mark on a deceptive line) is 59% (1704/2908),
computed with the Honeyquest rule (needs both a deceptive and a non-deceptive
mark). Drop `--group-by none` for per-model scorecards.

## Run it

```sh
uv venv && source .venv/bin/activate
uv pip install 'anthropic[bedrock]' pyyaml boto3   # baseline (Claude + Bedrock)

cp .env.example .env    # then fill in whichever auth you have
set -a; source .env; set +a

PYTHONPATH=src python src/run.py --model sonnet-4-6 --query-set canonical-30 --trials 3
```

Output: `results/{your-handle}_sonnet-4-6_canonical-30_3t_{ts}.jsonl` plus a
`.manifest.json` sidecar.

`--query-set` takes `canonical-30`, `full`, or a path to your own JSON.
Run `python src/run.py --help` for the model aliases and all flags.

## Local models (optional)

Most aliases are hosted (Anthropic API or AWS Bedrock) and download nothing.
Only two run on your machine, and each pulls its weights on first use:

`whiterabbit-neo-8b` — HuggingFace transformers. On the first run it downloads
`WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B` (~16 GB in fp16) from the HF Hub
and caches it under `HF_HOME` (default `~/.cache/huggingface`). No HF token
needed; the weights are ungated. Default device is `mps` (Apple Silicon); edit
the runner for CUDA/CPU.

```sh
uv pip install torch transformers
# optional: point the cache somewhere with room
export HF_HOME=/path/with/disk
PYTHONPATH=src python src/run.py --model whiterabbit-neo-8b --query-set canonical-30 --trials 1
```

`foundation-sec-8b` — runs through [Ollama](https://ollama.com), not transformers.
Install and start Ollama first; the GGUF model
(`hf.co/fdtn-ai/Foundation-Sec-8B-Reasoning-Q8_0-GGUF`) is pulled on first use.

```sh
# install Ollama, then:
ollama serve &
PYTHONPATH=src python src/run.py --model foundation-sec-8b --query-set canonical-30 --trials 1
```

Everything else (`llama4-maverick-17b`, `deepseek-*`, `gpt-oss-*`, `qwen3-*`,
`kimi-*`, etc.) is Bedrock-hosted — no local download.

## Note on the YAML query DB

The run path reads the baked JSON in `data/`, so you do NOT need the upstream
Honeyquest YAML repo. `loader.iter_queries()` only matters if you regenerate
those JSONs after an upstream refresh; point it at the YAML with
`export HONEYQUEST_ROOT=/path/to/honeyquest/querydb/queries`.
