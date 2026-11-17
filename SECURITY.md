# Security policy

## Reporting a vulnerability in this tool

Report vulnerabilities in this repository privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. Include the version (`llm-bench --version`),
a description of the problem and, if you can, steps to reproduce. Do not open a public issue for
a security problem, and never paste API keys or unredacted result files into an issue.

Credential leaks are in scope. That includes a key or other secret appearing in terminal output,
a log line or a result file.

## Problems with a provider's API

Problems with a provider's hosted API or your account belong with that provider. For the MolochAI
API, contact MolochAI at
[molochai.dev/contact](https://molochai.dev/contact?utm_source=github&utm_medium=oss&utm_campaign=llm-api-benchmark).

## API keys

`llm-bench` reads keys only from environment variables and never writes them to result files. If
a key has been exposed anyway, revoke it with its provider and create a new one. For MolochAI that
is [dash.molochai.dev/account](https://dash.molochai.dev/account?utm_source=github&utm_medium=oss&utm_campaign=llm-api-benchmark).

## Supported versions

Security fixes go into the latest release.
