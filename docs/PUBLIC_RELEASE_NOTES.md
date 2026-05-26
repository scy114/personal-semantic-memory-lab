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
