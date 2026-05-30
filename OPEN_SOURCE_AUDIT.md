# Open Source Audit

Status: local public mirror prepared, tested, and audited for initial review.

Audit date: 2026-05-31

## Can Publish

- Source code under `tools/`.
- Public configs under `configs/`.
- Unit tests and small fixtures under `tests/` and `fixtures/`.
- Public-safe docs under `docs/`.
- `.env.example`, README, Apache-2.0 license, CI workflow.

## Validation

- Local tests passed:
  `python -m unittest discover -s tests`
- Result: `Ran 282 tests in 12.179s ... OK`
- CI workflow uses the same unittest command.
- Live provider tests are not enabled by default.

## Excluded

- `users/`
- `data/`
- `reports/`
- `experiments/`
- `review_artifacts/`
- `external_references/`
- `.env`
- local model caches and vector/database outputs

## Secret Risk

No real API key was found by local grep audit. Placeholder environment variable
names such as `OPENAI_API_KEY` may appear in examples and code. Live calls are
disabled by default and require `ALLOW_LIVE_API=true`.

The 2026-05-31 audit found only expected code/example references for API-key
loading and live authorization text. No concrete provider secret or private
provider URL was found.

Checked categories included:

- private provider markers and private provider URLs;
- live authorization fragments: `Bearer ...`, long `sk-...` style keys;
- private key files: `.env`, `*.pem`, `*.key`, `id_rsa`, `id_ed25519`.

## License Unknown

Downloaded external datasets, lexicons, and reference repositories are excluded
from this mirror. Their licenses must be checked before redistribution.

## Nested Git Resolution

The private workspace used `tools/` as a nested git repository. The public mirror
copies it as a normal source directory and removes nested `.git` metadata.

## Large File Risk

Generated vectors, databases, model caches, and experiment outputs are excluded
by `.gitignore`. Local large-file audit found no file larger than 5 MB.

## Private Directory Audit

After cleanup, the mirror contains no `users/`, `data/`, `reports/`,
`experiments/`, `review_artifacts/`, `external_references/`, nested `.git`,
`__pycache__`, or `.pytest_cache` directories.

Some tests create temporary ignored output directories such as `users/` while
running. These are runtime artifacts only and are excluded by `.gitignore`.
