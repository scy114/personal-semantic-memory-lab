# Security Policy

## Reporting

Please report suspected security issues privately to the repository maintainer.
Do not open public issues containing secrets, API keys, private data, or exploit
details.

## Secrets

Do not commit:

- `.env` files;
- API keys or provider tokens;
- private user workspaces;
- live provider outputs;
- model caches or downloaded datasets.

The public test suite must run without live provider access.
