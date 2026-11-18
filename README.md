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

`--mode openevolve` runs [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) (0.4.1,
pinned) as an arm; `--mode harnesses` runs it with `direct` and `opencode`. OpenEvolve's own loop
(5 MAP-Elites islands, diff edits, top and diverse programs in the prompt) picks parents and builds
prompts. Three things are shared with the other arms, so the comparison is fair:

- Model calls go through the direct arm's `call_llm` (same `DIRECT_MODEL`, reasoning effort, pricing
  and retries).
- Candidates are scored by the same evaluator. The parent's per-scale diagnostics appear in the
  prompt as an artifact.
- Each iteration is recorded as `candidates/openevolve/NNN/`, so `holdout`, `analysis` and
  `signals` read it unchanged.

The budget rule is the same, checked before each model call. One worker runs iterations one at a
time. The task text and objective go in OpenEvolve's system message. OpenEvolve's logs,
checkpoints and settings are in `<run>/openevolve/`. Resuming isn't supported.

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

`early_signal/` gets `sessions.csv`, `summary.json` and four plots, each titled with a plain-language
verdict: final score vs the best score among the first k self-tests (A); the share of variation in
final score explained at each checkpoint, using only sessions with at least 3 more self-tests to go
and at least 20 such sessions (B); final score vs lines changed before the first self-test (D); and
final score vs the number of self-tests, which is known only at the end (E). There is no plot C: a
ratio of final to early score plotted against the early score correlates by construction when
finals barely vary. Sessions that OpenCode ended by auto-rejecting a
permission request (the agent reached outside its workspace; see `stderr.txt`) are excluded as cut off.
`./evaluate` copies each tested version and result into the workspace as `./evals/<n>/`.

# Self-test ablations

Two options (`--mode opencode` only) take a capability away from the agent's `./evaluate`. Every
other setting stays the same, so results compare against an unablated session run from the same
start:

- `--self-test-feedback score`: the agent sees only eligibility and the aggregate score, both from
  `./evaluate` (and in its `./evals/<n>/eval.json`) and for the parent in its prompt. If a scale
  fails, its error is still shown, so a broken scheduler can be fixed. The archive in
  `internal_evals/` keeps the full result for analysis.
- `--max-self-tests N`: `./evaluate` refuses (exit 3, nothing evaluated) after N runs in a session.
  The count is kept in `internal_evals/`, outside the agent's workspace.

Both are recorded in `run_config.json` under `self_test`, and per session in `command.json`.

```sh
# 10 sessions each from the seed on hard, against runs/sessions-hard-main (40, unablated)
uv run python -m src.sweep --mode opencode --repeats 10 --jobs 3 --traces scenarios/train/hard.csv \
  --model openai/gpt-6.1-sol --output runs/ablation-score-only -- --iterations 1 --agent-timeout 7200 \
  --self-test-feedback score
uv run python -m src.sweep --mode opencode --repeats 10 --jobs 3 --traces scenarios/train/hard.csv \
  --model openai/gpt-6.1-sol --output runs/ablation-3-tests -- --iterations 1 --agent-timeout 7200 \
  --max-self-tests 3
```

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
