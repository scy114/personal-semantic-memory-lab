# Public Architecture Overview

The project organizes memory construction into evidence-bound layers:

```text
S0B source/text organization
  -> route_s1
  -> S1 memory candidate build/index
  -> route_s2
  -> S2 portrait/user-model build/index
  -> graph construction
  -> graph algorithms, retrieval, visualization review
  -> incremental maintenance
```

## Boundaries

- S0B organizes source text and preserves backpointers.
- S1 produces evidence-bound memory substrate.
- S2 produces proposal-backed user-model units.
- Graph construction produces candidate nodes, edges, claims, and evidence
  links.
- Query uses lexical, semantic, and graph branches where available.
- Incremental maintenance records operations, resolves impact, publishes latest
  views, and plans partial refresh.

## Non-Truth Rules

- LLM output is candidate material unless reviewed.
- Graph candidates are not graph truth.
- Graph metrics are retrieval/ranking signals, not support proof.
- Visualization is for audit, not correctness proof.
