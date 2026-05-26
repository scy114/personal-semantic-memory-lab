# Personal Semantic Memory Lab

Personal Semantic Memory Lab is a research-oriented local-first toolkit for
evidence-bound personal memory, routed LLM assistance, subject graph
construction, graph-aware retrieval, visualization review, and incremental
maintenance.

This repository is a public mirror. It contains code, configs, tests, public
fixtures, and public-safe documentation. It does not contain private user data,
live provider outputs, downloaded external datasets, model caches, or local
experiment workspaces.

## What This Is

- A lab for building personal semantic memory systems.
- A workflow toolkit for S0B/S1/S2 memory construction and indexing.
- A graph construction and graph-aware retrieval prototype.
- A maintenance layer for append-only operation logs, stale detection, latest
  views, and incremental refresh planning.

## What This Is Not

- Not a finished consumer application.
- Not a durable memory service out of the box.
- Not a graph truth system.
- Not a support-checking authority.
- Not a hosted provider integration bundle.

Every graph candidate and graph metric should be treated as an audit/retrieval
signal, not proof.

## Current Public Baseline

- `v0.21`: pre-build routed, LLM-assisted S1/S2 build and index workflow.
- `v0.3`: evidence-bound graph construction, NetworkX utility checks, and
  graph-aware retrieval.
- `v0.31`: graph visualization review as an auxiliary audit track.
- `v0.4`: incremental maintenance skeleton, including operation log, impact
  resolver, latest views, S0B batch registration, S1/S2/graph affected reports,
  and incremental-vs-full-observation comparison.

## Repository Layout

```text
tools/      Core runners and helpers.
configs/    Prompt, proposal, routing, and graph configs.
tests/      Unit tests and small public fixtures.
fixtures/   Public fixture inputs.
docs/       Public-safe architecture and release notes.
```

Private/generated directories intentionally excluded from this public mirror:

```text
users/
data/
reports/
experiments/
review_artifacts/
external_references/
.env
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[test]
python -m unittest discover -s tests
```

The test suite is designed to run without live LLM calls. Provider-backed
experiments require explicit configuration and are not part of default CI.

## Provider Configuration

The default provider mode is mock/replay-oriented. To use a real
OpenAI-compatible provider, create a local `.env` from `.env.example` and set:

```text
OPENAI_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
ALLOW_LIVE_API=true
```

Do not commit `.env` files or provider outputs.

## External Resources

External datasets, lexicons, GraphRAG-style reference repositories, local
embedding models, and downloaded NLP resources are not vendored here. Use the
public docs and configs as integration references, and download external
resources under their original licenses.

## License

Code in this public mirror is released under the Apache License 2.0. External
resources referenced by the project retain their original licenses and are not
covered by this repository license unless explicitly included.
