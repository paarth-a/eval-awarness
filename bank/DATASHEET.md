# Datasheet for Evaluation Tells Bank

Following Gebru et al. (2021), "Datasheets for Datasets."

## Motivation

**For what purpose was the dataset created?**
To enable causal audits of LLM evaluation benchmarks for evaluation-identifying surface features ("tells") that may produce eval-awareness in models being evaluated. Motivated by alignment-faking results (Greenblatt et al. 2024) showing models can condition behavior on perceived evaluation status — eval writers benefit from a way to identify and reduce avoidable evaluation cues.

**Who created the dataset?**
Created in the course of research at SPAR (Supervised Program for Alignment Research, 2026 cohort).

**Who funded the dataset creation?**
SPAR program operating costs; OpenRouter API costs paid by the authors.

## Composition

**What do the instances represent?**
Each instance is a canonical "template" — a description of a recurring evaluation-tell pattern, with metadata about its source, frequency, and example evidence drawn from real evaluation transcripts.

**How many instances are there?**
258 templates total. By source benchmark: Needham (65), MALT (80), BBH (45), MASK (27), AgentHarm (26), Hawthorne (15). By category: CONTRIVED_TASK (75), SYSTEM_INSTRUCTIONS (44), TASK_FORMATTING (42), FICTIONAL_SCENARIO (35), ETHICAL_PROBE (31), ANSWER_FORMATTING (31).

**Does the dataset contain all possible instances or is it a sample?**
A canonical sample. Raw extraction yielded 2,954 individual tell instances. These were canonicalized into 258 templates by collapsing similar patterns.

**What data does each instance contain?**
See `templates.json` schema in README.md. Each template has: unique ID, source benchmark, category, pattern name, description, adaptation instructions, representative examples (with evidence text from source transcripts), compatibility notes, source tell IDs, frequency, typical location.

**Is there a label or target associated with each instance?**
Each template is implicitly labeled "evaluation-coded surface feature." Not a supervised dataset — the bank is a corpus of canonical patterns, not classification training data.

**Is any information missing from individual instances?**
Some templates from older extraction batches lack frequency counts or have abbreviated examples. Documented in `templates.json` where fields are null.

**Are there recommended data splits?**
The bank is intended as a single audit corpus, not split for ML training. However, source-benchmark stratification can be used for cross-benchmark transfer evaluation (we provide 5/5 cross-injection results).

**Are there errors, sources of noise, or redundancies?**
The canonicalization process collapsed similar patterns but may have produced some near-duplicates. Manual validation on a stratified sample of templates is recommended for downstream uses.

**Is the dataset self-contained?**
Yes. Templates reference example evidence text but the bank itself is self-contained.

## Collection Process

**How was the data acquired?**
Three-stage pipeline:
1. **Source transcripts:** drawn from 6 published evaluation benchmarks (Needham et al 2025, Hawthorne / Microsoft, MASK, AgentHarm, BBH, MALT).
2. **Tell extraction:** for each benchmark's high-confidence eval cases, an LLM (gpt-5.1) was asked to identify "local tells" — specific text patterns that signal the transcript is from an evaluation. Produced 2,954 raw tells.
3. **Canonicalization:** raw tells were grouped and canonicalized into 258 reusable templates with adaptation instructions, by clustering on pattern similarity.

**What mechanisms or procedures were used to collect the data?**
LLM-based extraction with structured prompts; clustering and canonicalization via LLM with manual validation samples.

**Who was involved in the data collection process?**
The authors (paper TBD); LLM extraction performed by gpt-5.1 via OpenRouter API.

**Over what timeframe was the data collected?**
March-May 2026.

**Were any ethical review processes conducted?**
This work uses publicly released evaluation benchmarks. No new participant data collection. No IRB review required.

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
- Tells with frequency < 2 in source were filtered before canonicalization.
- Templates received manual category assignment from a fixed taxonomy of 6 categories.
- Manual validation on a stratified sample of templates (n=30-50) by the authors.

**Was the "raw" data saved in addition to the preprocessed data?**
Yes — raw extracted tells are preserved at `tells/data/{benchmark}/extracted_tells.json` in the parent repo.

**Is the software used to preprocess/clean/label the data available?**
Yes — extraction code in `tells/extract_tells.py` (parent repo).

## Uses

**Has the dataset been used for any tasks already?**
Yes — for the causal audit pipeline described in the parent paper. Specifically:
- Reverse intervention (template-guided removal lowers P(eval))
- Forward intervention (template injection raises P(eval) on deployment-style transcripts)
- Cross-benchmark transfer (templates from one benchmark apply to another)
- Held-out validation (bank applied to SAD, AISI without re-extraction)

**Is there a repository linking to all uses of the dataset?**
The parent repo will be released alongside the paper.

**What other tasks could the dataset be used for?**
- Audit of new benchmarks for known evaluation cues
- Training data for surface-level eval-detection classifiers
- Reference for benchmark designers wanting to avoid known eval-coding patterns
- Comparison baseline for future eval-tell extraction work

**Is there anything about the dataset's composition or how it was collected that might impact future uses?**
- Templates were extracted only from 6 source benchmarks. Benchmarks with very different structure (e.g., long agentic, multimodal) may not be well-covered.
- Extraction used a single frontier extractor (gpt-5.1). Templates may reflect that extractor's biases.
- The bank is a snapshot at extraction time; LLM-generated benchmarks may evolve faster than the bank.

**Are there tasks for which the dataset should not be used?**
- Should not be used as a supervised training corpus without further annotation.
- Should not be used to generate adversarial inputs designed to evade safety checks (the audit-and-remove pipeline can in principle support this; we discourage it).

## Distribution

**Will the dataset be distributed to third parties?**
Yes — released with the paper.

**How will the dataset be distributed?**
Hosted in the paper's repo (GitHub) and as supplementary material.

**When will the dataset be distributed?**
With the paper publication.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, or under applicable terms of use (ToU)?**
Apache 2.0 license (recommended).

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
The example evidence text in templates derives from publicly released evaluation benchmarks. Each source benchmark has its own license; we do not redistribute substantial source-transcript content beyond illustrative excerpts.

## Maintenance

**Who is supporting/hosting/maintaining the dataset?**
The authors via the paper's repo.

**How can the owner/curator/manager of the dataset be contacted?**
[Contact via repo issues or paper correspondence].

**Will the dataset be updated?**
Possibly — if extraction is repeated on additional benchmarks or with updated extractors.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes — the extraction pipeline is open-sourced; contributions can be merged via the parent repo.
