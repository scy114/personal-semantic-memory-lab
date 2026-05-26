# S1 Memory Candidate Strong Proposal Prompt v0.1

Return strict JSON only. Do not include Markdown.

Task: produce an evidence-bound S1 memory candidate proposal from the input packet when simple script processing is insufficient.

Use the local text and provided context to compress complex statements, multi-clause events, preferences, procedures, constraints, or source-bound summaries. The output is proposal material only.

Rules:
- Keep S1 evidence discipline strict.
- The original text, deterministic processing result, evidence refs, and raw backpointers are runner-owned and must travel with your auxiliary output.
- Do not create or modify evidence refs.
- Do not write durable `memory_units.jsonl`, reviewed portrait units, current portrait, graph truth, graph nodes, graph edges, or support status.
- Do not turn the source into a portrait trait or graph relation.
- Prefer a concise factual memory candidate when evidence is explicit or directly inferable.
- Use `model_uncertain` when the text may contain useful S1 material but the compression or attribution is too ambiguous.
- Use `needs_human_review` when source identity, privacy, attribution, or provenance risk makes automatic proposal unsafe.
- Use `reject` only when there is no useful S1 memory candidate.

If `output_kind` is `reject`, `model_uncertain`, or `needs_human_review`:
- `memory_candidate_text` must be ""
- `memory_class` must be "unknown"

Required JSON keys:
{
  "output_kind": "memory_candidate|reject|model_uncertain|needs_human_review",
  "memory_candidate_text": "",
  "memory_class": "semantic|episodic|procedural|summary|unknown",
  "source_text_quote": "",
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
