# Development and releases

## Local development

The README explains environment setup, tests, and local builds. [Development](https://github.com/ZaberKo/torch-prisma/blob/main/README.md#development)

## Read the Docs

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
