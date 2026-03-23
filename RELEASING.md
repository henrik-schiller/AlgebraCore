# Releasing AlgebraCore

`AlgebraCore` is set up for GitHub-based PyPI publishing with trusted
publishing.

## One-time PyPI setup

Before the first PyPI release, create a trusted publisher for the project on
PyPI.

- PyPI project name: `AlgebraCore`
- Owner: `henrik-schiller`
- Repository: `AlgebraCore`
- Workflow file: `.github/workflows/publish.yml`
- Environment name: `pypi`

If the project does not exist on PyPI yet, create it there while adding the
trusted publisher.

## Release flow

1. Make sure `version` in `pyproject.toml` is the release version.
2. Push the release commit to `main`.
3. Create a GitHub release, for example `v0.1.0`.
4. Publishing runs automatically via `.github/workflows/publish.yml`.

## Notes

- The package is built from source in GitHub Actions.
- PyPI authentication is handled by trusted publishing, not by storing a PyPI
  API token in the repository.
- Local installation from GitHub remains useful before the first PyPI release:
  `pip install "git+https://github.com/henrik-schiller/AlgebraCore.git"`
