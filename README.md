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

Simple / Medium / Hard inputs are in `scenarios/`. Select one with
`--trace scenarios/train/simple.csv`. Validate the inputs with:

```sh
uv run python -m scenarios.make_workloads --validate-only
```

# Repeated runs

Run each train trace 5 times (the default), passing run options after `--`:

```sh
uv run python -m src.sweep --mode both --repeats 5 -- \
  --budget-usd 2 --budget-guard --budget-slack-usd 0.10 --iterations 1000
```

Each run gets its own directory, `runs/sweep-<time>/<split>-<level>/rep<k>/`. Re-running
with the same `--output` skips runs that already have results.

Score each run's best-so-far schedulers on the matching held-out test trace (no model calls),
then rebuild the sweep report:

```sh
uv run python -m src.holdout runs/sweep-<time>/*/rep*
uv run python -m src.analysis --sweep runs/sweep-<time>
```

# Tests

```sh
uv run python -m unittest discover -s tests -v
```
