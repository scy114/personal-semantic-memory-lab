# Step 1 Toolbox Contract Fixtures

Date: 2026-05-17

Purpose:

These fixtures validate `docs/step1-query-toolbox-contract.md`.

They are not production data, not benchmark results, and not canonical memory.

The fixture set covers:

- raw evidence retrieval;
- memory-unit retrieval with default accepted status;
- degraded retrieval warnings;
- valid and unresolved ref resolution;
- direct / unknown / contradicts support checks;
- grounding reports;
- Step1-to-Step2 handoff packet in by-ref and by-value modes.

## Files

```text
source_manifest.jsonl
evidence.jsonl
memory_units.jsonl
retrieval_results.valid.jsonl
retrieval_results.degraded.jsonl
resolved_refs.valid.jsonl
resolved_refs.degraded.jsonl
support_checks.jsonl
grounding_reports.jsonl
handoff_packet.by_ref.json
handoff_packet.by_value.json
```

## Validation Expectations

- JSON / JSONL must parse.
- Every contract object must include `schema_version`.
- Retrieval results must include `warnings`.
- Default `retrieve_evidence()` behavior should return raw evidence only.
- Optional layers must include warnings.
- `check_claim_support()` should trust raw evidence refs by default.
- `handoff_packet.by_ref.json` should avoid duplicating full evidence text.
- `handoff_packet.by_value.json` may embed compact text for debugging, but must keep canonical refs.
