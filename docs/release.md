# Development and releases

[Local development](https://github.com/ZaberKo/torch-prisma/blob/main/README.md#development)

## Accelerator validation

Run the **Accelerator validation** workflow with `cuda` or `ascend` on a configured self-hosted runner. The Ascend runner must have CANN initialized and TorchNPU, TorchAir, and pytest available to `python3`.

## Read the Docs

Connect GitHub to Read the Docs, add `ZaberKo/torch-prisma`, and select `main` as the default branch. The repository's `.readthedocs.yaml` builds the Zensical site. Set `site_url` in `zensical.toml` once the canonical URL is assigned. [Add a project](https://docs.readthedocs.com/platform/latest/intro/add-project.html)

## Publishing

1. Configure the GitHub Actions environment `pypi` and register a Trusted Publisher for the `torch-prisma` PyPI project. Use repository `ZaberKo/torch-prisma`, workflow `publish.yml`, and environment `pypi`. [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
2. Update the package version and `uv.lock` as needed. Confirm that CI passes on the release commit.
3. Publish a GitHub release with the matching tag, such as `v0.1.0`. The workflow publishes the checked distributions to PyPI and attaches them to the GitHub release.
