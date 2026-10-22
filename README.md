# Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [OpenCode](https://opencode.ai/).

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
