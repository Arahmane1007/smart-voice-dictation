# Contributing

Thanks for your interest! This project uses [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync --all-packages
```

## Before opening a pull request

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

Slow tests (they download a real Whisper model) are skipped by default: `uv run pytest -m slow`.

Optional: install the git hooks with `uvx pre-commit install`.

## Conventions

- Code, comments and docs are in English.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`…).
- Never commit voice recordings, API keys or `.env` files.
