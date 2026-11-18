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

## synthetic-tools/: synthetic fixture data, tool calling

The same files for `benchmarks/synthetic-tools.yaml`: the `tools-v1` dataset against two simulated
endpoints. The simulator does not read prompts. It picks a tool by a hash of the messages, now and
then picks another or answers in text, and fills arguments from the schema with placeholder
values. **Most argument checks fail by design.** The files show what the tool-call tables and
fields look like, not how any model calls tools.

| File | Made with |
|---|---|
| [terminal.txt](synthetic-tools/terminal.txt) | `llm-bench run --config benchmarks/synthetic-tools.yaml --yes --quiet --formats json,csv,md,html` (stdout) |
| [results.json](synthetic-tools/results.json), [requests.csv](synthetic-tools/requests.csv), [summary.csv](synthetic-tools/summary.csv) | the same run; see `tool_call_names` and `tool_signature` per request |
| [report.md](synthetic-tools/report.md), [report.html](synthetic-tools/report.html) | the same run |

## synthetic-conformance/: synthetic fixture data, conformance suite

`llm-bench conformance --config benchmarks/synthetic-conformance.yaml --formats json,md,junit`
runs the in-process reference endpoint twice: once as documented, and once with six quirks
switched on (ignored `max_tokens`, `response_format` and `n`; bare error bodies; an unknown model
served anyway; no usage in streams). **These are not results for any provider.** They show how
each kind of problem is reported.

| File | Contents |
|---|---|
| [terminal.txt](synthetic-conformance/terminal.txt) | stdout: the matrix and the reasons for each WARN and FAIL |
| [results.json](synthetic-conformance/results.json) | every test's outcome, summary, notes, status and latency |
| [report.md](synthetic-conformance/report.md) | the Markdown report |
| [results.junit.xml](synthetic-conformance/results.junit.xml) | JUnit XML for CI |
