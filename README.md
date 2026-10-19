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

Simple / Medium / Hard inputs are in `scenarios/complexity-full/`. Select one with
`--trace scenarios/complexity-full/train/simple.csv`. Validate the inputs with:

```sh
uv run python -m scenarios.make_workloads --validate-only
```

# Tests

```sh
uv run python -m unittest discover -s tests -v
```
