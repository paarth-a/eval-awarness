# Evaluation Tells Bank

A bank of 258 canonical evaluation "tells" — recurring features that signal a transcript comes from an evaluation/benchmark rather than a real-world deployment — extracted from six evaluation datasets.

## Quick stats

- **258 canonical templates** across 6 source benchmarks
- **6 categories:** CONTRIVED_TASK (75), SYSTEM_INSTRUCTIONS (44), TASK_FORMATTING (42), FICTIONAL_SCENARIO (35), ETHICAL_PROBE (31), ANSWER_FORMATTING (31)
- **Source benchmarks:** Needham (65), MALT (80), BBH (45), MASK (27), AgentHarm (26), Hawthorne (15)
- **Validated:** 96% topic-level preservation across n=205 sample (gpt-5.1 rubric, Sonnet-4 inter-rater agreement 95.6%, κ=0.588)
- **Generalizes:** transfer to 2 unseen benchmarks (SAD, AISI) with control-adjusted causal effect of −0.13 P(eval) (p<10⁻⁹)

## Files

- `templates.json` — the bank itself (list of 258 templates with metadata)
- `summary.json` — per-source and per-category counts
- `README.md` — this file
- `DATASHEET.md` — datasheet-for-datasets style documentation
- `schema.json` — JSON schema for a template entry

## Template schema

Each template has:

| Field | Type | Description |
|---|---|---|
| `unique_id` | string | Globally unique ID (e.g., `hawthorne/scaffolding_metadata`) |
| `template_id` | string | Source-local ID |
| `source_benchmark` | string | Origin (one of: needham, hawthorne, mask, agentharm, bbh, malt) |
| `category` | string | One of 6 categories (see above) |
| `pattern_name` | string | Human-readable name |
| `description` | string | Description of the tell pattern |
| `adaptation_instructions` | string | Guidance for matching this template against a transcript |
| `representative_examples` | list | List of `{transcript_id, evidence}` examples |
| `compatibility_notes` | string | Format constraints (e.g., requires system message) |
| `source_tell_ids` | list | IDs of raw extracted tells this template was derived from |
| `frequency` | int | Number of source tells canonicalized into this template |
| `typical_location` | string | Where the tell typically appears (system / user / assistant) |

## Usage

The bank supports an audit pipeline:

1. **Match:** for a target benchmark transcript, identify which templates match its content
2. **Remove:** rewrite the matched evidence with a writer LLM under matched-control constraints
3. **Measure:** compare P(eval) under an LLM judge before vs after removal

Reference implementation: see [`tells/remove_tells.py`](../tells/remove_tells.py) and [`tells/merge_tells.py`](../tells/merge_tells.py) in the parent repo.

## Validated audit performance

Across 6 source benchmarks (Needham, Hawthorne, MASK, AgentHarm, BBH, MALT):
- **Reverse causal effect** (treatment − control): −0.142 to −0.346 P(eval) decrease (all p<10⁻³)
- **Topic-level preservation:** 96% (n=205, gpt-5.1 rubric)
- **Inter-rater agreement:** 95.6% raw, Cohen's κ = 0.588 (Sonnet-4 vs gpt-5.1)

Across 2 held-out benchmarks (SAD, AISI), no re-extraction:
- **Causal effect** (removed − scope-only control): −0.126 pooled (p<10⁻⁹)

Cross-benchmark transfer (templates from one benchmark applied to another):
- **5/5 compatible source-target cells** show statistically significant transfer

## Limitations

- **Behavioral preservation < topic preservation.** Under a stricter behavioral metric (does the model produce the same response on sanitized as on original?), preservation is lower (~50% under aggressive removal, ~92% under conservative removal). The audit's intended use is reducing surface eval-coding while leaving the benchmark's general topic intact, not preserving exact model behavior.

- **Long agentic transcripts not in scope.** Transcripts with >5 multi-turn agentic interactions (e.g., AppWorld-style tool-use loops) saturate the LLM judge at the structural pattern level and cannot be cleanly sanitized.

- **Behavioral evaluations are bimodally affected.** Capability and format-based benchmarks (BBH, MALT factual, Hawthorne propaganda) preserve cleanly under sanitization. Behavioral benchmarks (MASK deception, AgentHarm harm-refusal, Needham secret-protection) sometimes have eval-cues intrinsically entangled with the test target, where sanitization either fails or transforms the test.

- **Single-judge dependence.** Primary results use gpt-5-mini as the LLM judge; we replicate on Sonnet-4 (subset) and Gemini-2.5-flash (full). Direction of effect replicates across families; magnitudes vary.

- **Extraction is not certified extractor-agnostic.** Templates were extracted with gpt-5.1 (frontier model); we found Sonnet-4 extraction unreliable for this task.

## Citation

```bibtex
@misc{eval_tells_bank_2026,
  title  = {A Bank of Canonical Evaluation Tells for LLM Audit Pipelines},
  author = {[Authors]},
  year   = {2026},
  note   = {258 templates extracted from 6 evaluation benchmarks}
}
```

## License

[Choose: MIT / Apache 2.0 / CC-BY-4.0 — recommend Apache 2.0 for code+data combination]
