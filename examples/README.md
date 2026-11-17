# Examples

## synthetic-demo/: synthetic fixture data

Every file in [synthetic-demo/](synthetic-demo/) comes from the in-process simulator
(`benchmarks/synthetic-demo.yaml`): two simulated endpoints, one with a 5% injected error rate,
with every delay scaled down to a quarter so the demo finishes in seconds. **None of these numbers
describe a real API.** They show what each output format looks like.

| File | Made with |
|---|---|
| [terminal.txt](synthetic-demo/terminal.txt) | `llm-bench run --config benchmarks/synthetic-demo.yaml --yes --quiet --formats json,csv,md,html` (stdout) |
| [results.json](synthetic-demo/results.json) | the same run: every request, the summary, config, dataset and environment |
| [requests.csv](synthetic-demo/requests.csv), [summary.csv](synthetic-demo/summary.csv) | the same run |
| [report.md](synthetic-demo/report.md) | the same run, or `llm-bench report results.json --format markdown` |
| [report.html](synthetic-demo/report.html) | the same run, or `llm-bench report results.json --format html`. Download it and open it in a browser. |
| [compare.txt](synthetic-demo/compare.txt), [compare.md](synthetic-demo/compare.md) | `llm-bench compare results.json results.json --provider-a sim-steady --provider-b sim-variable` |

To produce your own:

```bash
llm-bench init synthetic-demo
llm-bench run --config synthetic-demo.yaml --yes --formats json,csv,md,html
```
