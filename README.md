# Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [OpenCode](https://opencode.ai/) 1.x
(validated with 1.19.3; 2.x removed `run --dir` and `--variant`). The harness uses `.tools/opencode`
if present, else `opencode` on `PATH`; with several installs, pin one:

```sh
printf '#!/bin/sh\nexec /path/to/opencode-1.x "$@"\n' > .tools/opencode && chmod +x .tools/opencode
```

```sh
uv sync --locked
cp -n .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

# Run

Check without model calls:

```sh
uv run python -m src.run --mode both --iterations 0 --prototype
```

Run both methods with a target of $2 each (spending may exceed the target):

```sh
uv run python -m src.run --mode both --budget-usd 2 \
  --budget-guard --budget-slack-usd 0.10 --iterations 1000
```

Results are saved under `runs/`.

`--mode all` adds a third arm, `opencode_noeval`: the same agent with no `./evaluate`, told not
to run the simulator. Both agent arms work in a temporary directory outside the repository.
Each attempt's `generation_usage.agent_access` records tool calls that touched evaluation code,
eudoxia, the workloads or the repository, plus how often the agent ran `./evaluate`.

Simple / Medium / Hard inputs are in `scenarios/`. Select one with
`--trace scenarios/train/simple.csv`. Validate the inputs with:

```sh
uv run python -m scenarios.make_workloads --validate-only
```

# Repeated runs

Run each train trace 3 times (the default), passing run options after `--`:

```sh
uv run python -m src.sweep --mode all --repeats 3 -- \
  --budget-usd 2 --budget-guard --budget-slack-usd 0.10 --iterations 1000
```

Each run gets its own directory, `runs/sweep-<time>/<split>-<level>/rep<k>/`. Re-running
with the same `--output` skips runs that already have results. `--jobs N` runs N at once; each
run's console output then goes to `sweep.log` in its directory. Keep N within your API rate
limits and CPU cores. `--model openai/gpt-6-luna` runs every arm on that model instead of the
`.env` models; each run prints its models when it starts.

Score each run's best-so-far schedulers on the matching held-out test trace (no model calls),
then rebuild the sweep report:

```sh
uv run python -m src.holdout runs/sweep-<time>/*/rep*
uv run python -m src.analysis --sweep runs/sweep-<time>
```

# Early-signal sessions

Does early self-test progress in an agent session predict the session's final score? Run many
independent agent + evaluator sessions from one fixed starting scheduler (`--iterations 1` = one
session per run). Starting schedulers in `scenarios/start_schedulers/`, all direct-call results from
the shallow sweep on `train/simple.csv` (each run keeps a frozen copy and its checksum):

| File | Score | vs seed | One self-test takes |
| --- | ---: | ---: | ---: |
| `simple_direct_180s.py` | 179.9 s | 0.25x | ~30 s |
| `simple_direct_211s.py` | 211.0 s | 0.30x | ~66 s |
| `simple_direct_221s.py` | 221.4 s | 0.31x | ~210 s |

```sh
# 3-session pilot, then the main run with --repeats 40 --output runs/sessions-main
uv run python -m src.sweep --mode opencode --repeats 3 --jobs 3 --traces scenarios/train/simple.csv \
  --model openai/gpt-6.1-sol --output runs/sessions-pilot -- --iterations 1 --agent-timeout 3600 \
  --start-scheduler scenarios/start_schedulers/simple_direct_180s.py
uv run python -m src.early_signal runs/sessions-pilot --k 1 5
```

`early_signal/` gets `sessions.csv`, `summary.json` and four plots: final score vs the best score
among the first k self-tests (not floored at the start) with a least-squares fit and ±1 residual SD
band (A), fit quality by checkpoint with bootstrap intervals (B), improvement after k (C), and final
score vs lines of code changed before the first self-test (D).

# RQ3 signals and predictions

Build a per-candidate table of predictors (code, parent diagnostics, agent behavior) and
outcomes (own score, improvement over parent, best descendant) for a run or a whole sweep:

```sh
uv run python -m src.signals runs/sweep-<time>
```

Ask a model to predict each candidate's score from the parent's code and results plus the
child's code, without seeing the child's evaluation. This runs after the search and spends
separately from it, so `--max-usd` is required. Rebuild the table afterwards to add `pred_*`:

```sh
uv run python -m src.predict runs/sweep-<time> --max-usd 1
uv run python -m src.signals runs/sweep-<time>
```

The judge defaults to `PREDICT_MODEL` (or `DIRECT_MODEL`). Use `--tag` to keep a second judge
model's predictions separate.

# Tests

```sh
uv run python -m unittest discover -s tests -v
```
