# Releasing

How a version gets from `main` to a GitHub release and, once approved, to PyPI. This is for
maintainers.

## What the release workflow does

[.github/workflows/release.yml](.github/workflows/release.yml) runs when a GitHub release is
published, or by hand as a dry run (build and checks only, nothing uploaded).

| Job | Runs when | Permissions | What it does |
|---|---|---|---|
| `build` | always | `contents: read` | Builds the sdist and wheel, runs `twine check --strict`, runs [.github/scripts/check_dist.py](.github/scripts/check_dist.py) (one wheel, one sdist, tag `v<version>`, no direct references), installs the wheel in a fresh virtual environment, checks `llm-bench --version` and runs the offline demo. |
| `github-release` | a release was published | `contents: write` | Attaches the sdist and wheel to the GitHub release. |
| `pypi` | a release was published and the repository variable `PUBLISH_TO_PYPI` is `true` | `id-token: write`, environment `pypi` | Publishes the files to PyPI with Trusted Publishing. No token is stored. |
| `verify-pypi` | after `pypi` | none | Installs `llm-api-benchmark==<version>` from PyPI and runs `llm-bench --version`. |

Without `PUBLISH_TO_PYPI=true`, a release goes to GitHub only.

## One-time setup before the first PyPI release

1. On PyPI, add a pending publisher: account menu → **Publishing** → **Add a new pending
   publisher** → GitHub, with:
   - project `llm-api-benchmark`
   - owner `Moloch-AI`
   - repository `llm-api-benchmark`
   - workflow `release.yml`
   - environment `pypi`

   A pending publisher does not reserve the name.
2. On GitHub, create the environment `pypi` (Settings → Environments). Optionally restrict it to
   tags `v*` and require a reviewer.
3. Set the repository variable `PUBLISH_TO_PYPI` to `true` only for the release that should go to
   PyPI.

## Cutting a release

1. Update `src/llm_api_benchmark/_version.py` and `CHANGELOG.md` on `main`; CI must be green.
2. Check the name is still free right before the first upload:
   `curl -s -o /dev/null -w "%{http_code}" https://pypi.org/pypi/llm-api-benchmark/json` should
   print `404`. After that, the project must be ours.
3. Publish a GitHub release with the tag `v<version>`, for example `v0.1.0`.
4. Follow the workflow. Once it is on PyPI, change the README's installation section to
   `pipx install llm-api-benchmark`.
