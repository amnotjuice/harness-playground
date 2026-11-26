# Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [OpenCode](https://opencode.ai/).

```sh
uv sync --locked
cp -n .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

# Run

One entry runs both methods from the same seed, prompt, evaluator and budget rule:

```sh
uv run python -m src.run --mode both --budget-usd 2
```

- Train on `scenarios/train/hard.csv` (default; change with `--trace`). After search,
  each arm's best training scheduler is evaluated once on `scenarios/test/hard.csv`.
- Every model request (Direct, selector and every OpenCode call) goes through a local
  proxy that bills it from the provider's token usage. Once an arm reaches
  `--budget-usd`, further requests are refused, also mid-session; the last request may
  overshoot. Unknown billing stops the run.
- Every attempt is kept. Byte-identical source is not added to the parent pool again.
- Results are saved under `runs/`.

Check without model calls:

```sh
uv run python -m src.run --mode both --iterations 0 --prototype
```

Validate the workloads:

```sh
uv run python -m scenarios.make_workloads --validate-only
```

# Tests

```sh
uv run python -m unittest discover -s tests -v
```
