# S1 Memory Candidate Weak Proposal Prompt v0.1

Return strict JSON only. Do not include Markdown.

Task: produce an S1 memory candidate proposal when the input text contains evidence-bound material worth preserving for later memory processing.

Rules:
- S1 is an evidence/provenance layer. Do not write durable memory units.
- LLM output is auxiliary. It must never replace the original text carried by the runner.
- Prefer deterministic/script processing when it is sufficient. Use this prompt only to assist compression or local extraction.
- Do not create, rename, or invent evidence refs, raw backpointers, graph nodes, graph edges, portrait facts, or S2/S3 claims.
- A memory candidate should be a concise factual compression of the source text, not a portrait interpretation.
- Keep the candidate tied to the source text and quote exact source text when possible.
- If attribution, source meaning, or memory value is unclear, use `model_uncertain` or `needs_human_review`.
- If there is no useful S1 memory material, use `reject`.

Allowed `output_kind`:
- `memory_candidate`
- `reject`
- `model_uncertain`
- `needs_human_review`

If `output_kind` is `reject`, `model_uncertain`, or `needs_human_review`:
- `memory_candidate_text` must be ""
- `memory_class` must be "unknown"

Required JSON keys:
{
  "output_kind": "...",
  "memory_candidate_text": "...",
  "memory_class": "semantic|episodic|procedural|summary|unknown",
  "source_text_quote": "...",
  "source_text_quotes": [],
  "supporting_observations": [],
  "scope_hint": "event|session|project|relationship|document|unknown",
  "temporal_hint": "current|historical|event_bound|unknown",
  "compression_level": "light|medium|unknown",
  "inference_level": "explicit|direct_inference|weak_inference|unknown",
  "proposal_confidence": "high|medium|low|unknown",
  "uncertainty_notes": [],
  "warnings": []
}
