# Public Release Notes

## v0.21

Closed the routed, LLM-assisted S1/S2 build-and-index workflow. Pre-build
routing became workflow policy. LLM-assisted S1 text remains paired with
original evidence and provenance.

## v0.3

Added evidence-bound graph construction, graph candidate consolidation,
NetworkX utility checks, and graph-aware retrieval. The graph layer remains
candidate-first and evidence-bound.

## v0.31

Added graph visualization review as an auxiliary audit track. Visualization is
not proof; it helps inspect clusters, noisy edges, entity merge errors, and
evidence neighborhoods.

## v0.4

Added incremental maintenance skeleton:

- operation log;
- impact resolver;
- latest view projection;
- S0B incremental batch registration;
- S1 delta comparison;
- S2 affected-unit resolver;
- graph affected-packet resolver;
- incremental-vs-full-observation comparator.

## v0.41

Added a thin unified workflow entrypoint:

```powershell
python -m tools.workflow_runner ...
```

The entrypoint orchestrates existing runners without replacing their logic:

- `status`: inspect S0B/S1/S2/graph/index/current-view assets;
- `build-full`: run S1 Build -> S1 Index -> S2 Build -> S2 Index;
- `build-graph`: run graph packets, routing, extraction, consolidation,
  NetworkX, profile/community, quality gate, and optional visual bundle;
- `query`: run Step 2 query with lexical / embedding / graph context wiring;
- `incremental`: prepare/finalize/smoke v0.4 human-review maintenance flows.

Acceptance status:

- mock/synthetic acceptance path passed in the private validation workspace;
- full private test suite passed before public sync;
- bounded live provider smoke remains explicitly gated;
- no durable memory write, graph truth write, S3, or support-checker authority
  is enabled by this release.
