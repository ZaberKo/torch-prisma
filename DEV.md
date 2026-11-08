# Development

Run the commands in this guide from the repository root. Use uv and Python 3.12 for development. The library supports Python 3.10 and later. Zensical and pre-commit 4.7 require Python 3.11 or later.

## Set up the environment

Install the package in editable mode and sync the locked dependencies.

```bash
uv sync --locked --python 3.12
```

For a CPU-only development environment, use the PyTorch CPU index during setup.

```bash
uv venv --python 3.12
uv pip install --torch-backend=cpu --group dev -e .
```

## Run checks

Use `--no-sync` to run the checks in the environment from the setup step.

```bash
uv run --no-sync pre-commit install --install-hooks
uv run --no-sync pre-commit run --all-files
uv run --no-sync python -I -m pytest
```

Pre-commit runs Ruff linting and formatting, validates common file formats, and checks whitespace and line endings. Use Google-style docstrings in the library. Review and stage any fixes the hooks apply. Declare compatible dependency ranges in `pyproject.toml` and refresh `uv.lock` with `uv lock` when changing dependencies. Update the Ruff dependency range and hook revision together.

GitHub Actions runs the hooks and the CPU compatibility matrix. The manual **Accelerator validation** workflow runs on a matching self-hosted runner. See [Ascend setup](docs/ascend.md) for its runtime requirements.

## Build distributions

```bash
uv build
uvx twine check --strict dist/*.whl dist/*.tar.gz
```

`uv build` uses `uv_build` to create a wheel and source distribution in `dist/`. Twine validates their package metadata and rendered description. See the [release guide](docs/release.md) for publishing.

## Build documentation

Edit the Markdown pages under `docs/`. Preview the site locally or build it with strict link validation.

```bash
uv run --locked --isolated --only-group docs zensical serve
uv run --locked --isolated --only-group docs zensical build --strict
```

Zensical builds the site into `site/` from the locked `docs` dependency group. GitHub CI runs the strict build. See the [Read the Docs setup](docs/release.md#read-the-docs) for hosting.
