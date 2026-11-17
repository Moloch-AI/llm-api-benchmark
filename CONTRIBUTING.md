# Contributing

Thanks for helping. Bug reports, provider configurations you have verified, dataset proposals and
fixes are all welcome.

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Moloch-AI/llm-api-benchmark
cd llm-api-benchmark
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Checks (the same as CI)

```bash
ruff check .
ruff format --check .
mypy
pytest
```

The unit tests run offline. They use in-process HTTP fixtures (`httpx.MockTransport`) and a real
HTTP server on localhost, so no key or network access is needed.

### Integration tests against the MolochAI mock server

`tests/integration/` runs the `molochai` preset against the official mock server, which needs
Node 20 or newer:

```bash
git clone https://github.com/Moloch-AI/uncensored-ai-api
node uncensored-ai-api/mock-server/server.mjs --port 4010 --quiet --rate-limit 600
# in another shell
MOLOCHAI_MOCK_BASE_URL=http://127.0.0.1:4010/api/v1 pytest tests/integration -v
```

Without `MOLOCHAI_MOCK_BASE_URL` these tests are skipped.

### Live check

`.github/workflows/live.yml` is started by hand only. Without the `MOLOCHAI_API_KEY` secret it
sends one request with an invalid key. With the secret it sends at most four short requests.
Never add scheduled or high-volume live benchmarks to CI.

## Guidelines

- **Measurements are not contributions.** Pull requests must not add benchmark results about real
  providers to the repository. The only sample output lives in `examples/`, and it is synthetic
  and labeled as such.
- **Provider configurations.** Add a provider to `benchmarks/` only if you have run it. Say in the
  pull request when and how you checked it, and keep it free of prices unless you cite the source
  and date in `pricing`.
- **Datasets.** Prompts must be harmless, written for this project or properly licensed, and free
  of personal data. Any change to a prompt, limit or check needs a new dataset `version`.
- **Templates.** `benchmarks/{synthetic-demo,molochai,openai-compatible}.yaml` must match
  `src/llm_api_benchmark/templates/` byte for byte, and a test enforces it.
- **Credentials.** Never commit keys, real tokens or `.env` files. Tests use obviously fake keys.
- **Code.** Keep runtime dependencies to `httpx` and `PyYAML`. Use typed code that passes
  `mypy --strict`, and add tests for new behavior.

## Releasing

See [RELEASING.md](RELEASING.md). Contributors do not need it.
