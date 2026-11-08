# Development and releases

## Local development

Install uv and run the following commands from the repository root. Use the repository's Python 3.12 environment for development tools. Pre-commit 4.7 requires Python 3.11 or later.

```bash
uv sync --locked
uv run --locked pre-commit install --install-hooks
uv run --locked pre-commit run --all-files
uv run --locked pytest
uv build
uvx twine check --strict dist/*
```

For a CPU-only test environment, select the PyTorch CPU index explicitly.

```bash
uv venv --python 3.12
uv pip install --torch-backend=cpu --group dev -e .
.venv/bin/python -m pytest
```

The hooks run Ruff linting and formatting and check common repository file formats and whitespace. Ruff enforces Google-style docstrings in the library. Review and stage any automatic fixes before committing. When updating Ruff, update its development dependency and hook revision together.

The regular GitHub Actions workflow runs the same hooks and the CPU test matrix and validates the wheel and source distribution. Accelerator validation runs on demand on a matching self-hosted runner. See [Ascend setup](ascend.md) for runner requirements.

## Documentation

The documentation site uses Zensical and the Markdown pages in `docs/`. Run these commands from the repository root to preview the site or build it with strict link validation.

```bash
uv run --locked --isolated --only-group docs zensical serve
uv run --locked --isolated --only-group docs zensical build --strict
```

Both commands use the locked documentation dependencies in an isolated environment. The build writes HTML to `site/`. GitHub CI runs the strict build.

### Read the Docs

1. Push the documentation configuration to the branch Read the Docs will build.
2. Connect your GitHub account in Read the Docs. If you use the GitHub App, grant it access to the repository.
3. Select **Add project**, choose `ZaberKo/torch-prisma`, and set the default branch to `main`. Read the Docs reads `.readthedocs.yaml` from the repository root.
4. Open **Builds** and check the first build of `latest`. Once the project URL is assigned, set `site_url` in `zensical.toml` to its canonical documentation URL.

The build configuration uses Python 3.12, installs uv, and builds the site from the locked `docs` dependency group. It copies the generated HTML into the Read the Docs output directory.

Importing a private repository requires Read the Docs Business. Read the Docs Community supports public repositories. [Project setup](https://docs.readthedocs.com/platform/latest/intro/add-project.html) · [Private repositories](https://docs.readthedocs.com/platform/latest/guides/creating-project-private-repository.html) · [Zensical deployment](https://docs.readthedocs.com/platform/latest/intro/zensical.html)

## Publishing version 0.1.0

1. Configure the repository's `pypi` GitHub Actions environment.
2. Register a pending Trusted Publisher on PyPI for the `torch-prisma` project. Set the GitHub owner and repository to the release repository, the workflow filename to `publish.yml`, and the environment to `pypi`. [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
3. Confirm that the release commit passes CI. Publish a GitHub release tagged `v0.1.0` at that commit.
4. Check the publish workflow. It verifies the tag against `pyproject.toml`, reruns CI, publishes the checked distributions through PyPI Trusted Publishing, and attaches the same files to the GitHub release.

For subsequent releases, update the version in `pyproject.toml`, refresh `uv.lock`, and use the matching tag.
