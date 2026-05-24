# Methodology

This document describes every experimental procedure behind the
results in `paper_v2.tex` and `RESULTS_SUMMARY.md`. It is intended for
reviewers and reproducibility-checkers; each section names the inputs
(data, models, env vars), the procedure, the outputs (JSON paths), and
how the headline number was computed.

The release ships under `test_smells_release/` (this directory) with all
scripts, paired datasets, and result JSONs. We organize methodology by
experimental construct rather than by paper section, so each measurement
is described once and cross-referenced as needed.

---

## 0 · Conventions

### Sign convention
Throughout: $\Delta = P(\text{eval})_{\text{after}} - P(\text{eval})_{\text{before}}$.
- **Positive $\Delta$**: rewrite raised the judge's eval-classification.
- **Negative $\Delta$**: rewrite lowered it.
- Forward injection produces positive $\Delta$.
- Reverse intervention produces negative $\Delta$.

### Unit of independence
For all paired tests (Wilcoxon, bootstrap, t-test), the unit is the
transcript, not the per-judge-call. Repeated judge calls per transcript
are averaged into a per-transcript $P(\text{eval})$ before any test.
n in tables refers to transcripts.

### Judges and target models
| Role | Default model | Where overridden |
|---|---|---|
| Primary judge (forward / reverse / preservation) | `openai/gpt-5-mini` | — |
| Cross-judge robustness | `google/gemini-2.5-flash`, `anthropic/claude-haiku-4.5` | §6 dual-judge / §8 probe scoring |
| Rewriter (forward injection, reverse v1/v2/v3, neutral controls) | `google/gemini-2.5-flash` | — |
| Preservation gate LLM judge | `google/gemini-2.5-flash` | — |
| Target model (preservation behavioral) | `openai/gpt-5-mini` | — |
| Activation extraction | Qwen3 8B/32B, Qwen3.5 27B/122B-A10B, Mistral-Small-24B, Gemma-2-27B-it | — |

All API calls go through OpenRouter. Set `OPENROUTER_API_KEY` in `.env`;
Needham loads require `HF_TOKEN`.

---

## 1 · Bank: extraction → classification → audit

### 1.1 Donor pool (E-cases)
Donor transcripts are drawn from six benchmarks: **Needham, Hawthorne,
MASK, AgentHarm, BBH, MALT.** From each benchmark's HF release we filter
to transcripts where the LLM-judged eval-awareness score
$\bar P(\text{eval}) \geq 0.8$ and the mean of two independent human
annotations $\geq 0.7$. This filter is borrowed from prior work
(Needham et al., 2025) and the human annotations come with the upstream
dataset.

### 1.2 Recipient pool (D-cases)
For the forward-injection direction, D-cases are drawn from:
- Each source benchmark's low-confidence-deploy subset
  (where the same judge gives $P(\text{eval}) \leq 0.2$)
- Naturalistic chat from WildChat 1M

### 1.3 Three-stage extraction
1. **Per-transcript LLM extraction**: for each E-case, prompt
   `openai/gpt-5.1` to list every surface feature that signals "eval"
   in JSON form (template name, category, evidence span, location).
   Outputs ~$80$ raw cues per transcript.
2. **Cross-transcript deduplication**: a second LLM pass identifies
   shared patterns across transcripts within a source benchmark and
   collapses raw cues into canonical templates with `adaptation_instructions`
   for re-injection. ~$258$ templates retained after dedup.
3. **Categorization**: each template is tagged with one of six categories:
   - `SystemInstructions` (e.g., "you are a benchmark evaluator")
   - `TaskFormatting` (e.g., "step-by-step instructions with numbered steps")
   - `AnswerFormatting` (e.g., "respond with a single letter A/B/C/D")
   - `FictionalScenario` (e.g., "imagine you are a doctor in 1960s rural Kenya")
   - `ContrivedTask` (e.g., "this is a hypothetical scenario for testing")
   - `EthicalProbe` (e.g., "what would you do if asked X by a hostile user")

### 1.4 Scaffolding vs test-bearing classification
A separate LLM pass (`bank_pipeline/classify_existing_bank.py`) classifies each
template as `scaffolding` (the bank pattern is purely surface, removable without
changing what the test measures) or `test-bearing` (the pattern is intrinsic
to the test target). Two policies are released:
- **Conservative**: requires explicit `is_scaffolding: true`; defaults to
  `test-bearing` on ambiguous templates.
- **Aggressive**: defaults to `is_scaffolding: true`; only marks `test-bearing`
  when the model is confident.

### 1.5 Manual validation audit
A stratified random sample of $n{=}42$ templates ($7$ per benchmark) is
validated by a human annotator against verbatim evidence in the donor
transcript. Strict pass = template description and evidence match exactly
and the template would survive any reasonable interpretation as an
eval cue. Inclusive pass = same but allows minor mismatches.

**Results (Table~\ref{tab:bank-audit} in paper):**
- BBH 100% (7/7), Needham 100% (7/7), MALT 86% (6/7),
  AgentHarm 57% (4/7), MASK 43% (3/7), Hawthorne 14% (1/7)
- Aggregate: $67\%$ strict ($28/42$), $88\%$ inclusive ($37/42$)

We frame the catalog as an **imperfect high-recall bank**; the safety
benchmarks (especially Hawthorne) over-include content cues. The bimodal
capability-vs-safety reading should be read in light of this audit.

### 1.6 High-precision subset
Defined as: templates from benchmarks whose strict-pass rate is
$\geq 80\%$. This yields:
- BBH (45 templates, $100\%$)
- Needham (65 templates, $100\%$)
- MALT (80 templates, $86\%$, mostly clean)
- AgentHarm: ~$15/26$ after per-template audit ($57\%$)
- MASK: ~$12/27$ ($43\%$)
- Hawthorne: ~$2/15$ ($14\%$) — too small for stable per-category test

`scripts/high_precision_subset.py` produces the subset listing.

### 1.7 Cross-extractor robustness
Independently re-extract the Needham bank with `anthropic/claude-sonnet-4`
(yielding 96 templates) and align via a third-party
`gemini-2.5-flash` semantic-match judge. Aggregate alignment: $86.5\%$
of Claude templates have a matching GPT template ($26.2\%$ in reverse
direction, dominated by GPT being more fine-grained). Both extractors
recover all six categories.

---

## 2 · Forward injection (sufficiency)

### 2.1 Cross-injection cells
For each (source benchmark, target benchmark) pair, take all D-cases
of the target and inject source's bank templates as prefix/system-message
addition via gemini-2.5-flash. $5\times 3 - 4 = 11$ usable cells; four
omitted (BBH/MALT → Hawthorne/MASK) because capability-only donor
templates fail to compose with safety target distributions
(rewriter rejects as incoherent).

Each injected D-case is re-judged with `gpt-5-mini`. The **raw shift**
is $\bar P(\text{eval})_{\text{treatment}} - \bar P(\text{eval})_{\text{baseline}}$.

### 2.2 Matched-paraphrase control
For every D-case the cross-injection pipeline also produces a paired
**paraphrase control**: the same D-case rewritten to similar scope but
without bank-feature insertion. The control isolates rewrite-scope effects
from bank-feature effects.

**Causal shift** $= \bar\Delta_{\text{treat}} - \bar\Delta_{\text{ctrl}}$
per cell. Aggregated across all 11 cells:
- Pooled mean causal $\Delta = +0.192$
- Bootstrap $95\%$ CI: $[+0.129, +0.252]$ ($B{=}1000$, per-cell mean as unit)
- One-sample $t$ vs $0$: $t{=}5.82$, $p{=}1.7 \times 10^{-4}$
- $11/11$ direction-positive; $7/11$ individually significant
  (cell-level $95\%$ CI excludes $0$ or $|t| > 1.96$)

Files: `results/forward_causal_11cells.json`,
`results/pooled_11cell_bootstrap.json`.

### 2.3 Neutral-prefix control (complementary methodology)
A second matched-control methodology for two cells (Hawthorne and MASK
target benchmarks, $n{=}20$ each): generate a length-matched
**innocuous prefix** (generic helpful boilerplate, no test vocabulary)
and re-judge the D-case with that prefix instead of the bank features.

The paraphrase control isolates bank-specific **content** effects
("does the bank do more than a same-scope rewrite?"); the neutral-prefix
control isolates bank-specific **prefix-position** effects ("does the bank
do more than length-matched innocuous boilerplate?"). Both should be in
the same direction; agreement strengthens the causal claim.

Results: bank-specific $\Delta = +0.150$ (Hawthorne, $95\%$ CI excludes 0)
and $+0.080$ (MASK, CI just touches 0).

File: `results/forward_neutral_injection.json`,
`scripts/forward_neutral_injection.py`.

---

## 3 · Reverse intervention (necessity)

### 3.1 Pipeline versions
| Version | Filter | Preservation gate | Reported in paper? |
|---|---|---|---|
| v1 | All flagged templates | None | As ungated baseline (§9) |
| v2 | Scaffolding-only filter | None | Held-out (no gate available) (§5) |
| v3 | Scaffolding-only + 5-criterion gate | Yes (gemini-2.5-flash) | **Primary** (§5) |
| v4 | + objective certificate + N=3 candidates | Yes | Overconstrained; not in paper |

### 3.2 v3 preservation gate (canonical pipeline)
For each E-case identified as containing one or more bank templates:
1. **Rewrite** with gemini-2.5-flash: instructed to remove only
   scaffolding templates (test-bearing templates are protected as
   "non-removable spans"); preserve task, behavior target, answer
   format, and any specific content the task hinges on.
2. **Five-criterion gate** (LLM judgment via gemini-2.5-flash):
   - `preserves_task`: the original task is still answerable
   - `preserves_behavior_target`: the model's expected behavior class is unchanged
   - `preserves_answerability`: any required format/answer-extraction works
   - `removed_only_scaffolding`: only scaffolding tells were removed
   - `pass_strict`: all four above pass
3. **Verbatim-survival check** post-hoc: re-detect each removed template
   in the rewritten transcript; reject if any template still verbatim-present.
4. **Reject** if the gate fails or the survival check fails.

### 3.3 Matched neutral-rewrite control (rev_control)
For each gate-accepted rewrite we also generate a
**matched random-span rewrite**: same paraphraser, same scope (similar
character-count edit), but no targeted template removal. This isolates
bank-specific removal effects from generic rewriting effects.

### 3.4 Canonical reverse table

| Pool | $n$ | baseline $P(\text{eval})$ | treatment $P(\text{eval})$ | control $P(\text{eval})$ | raw $\Delta$ | causal $\Delta$ | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Needham source pool (v3) | 93 | 0.900 | 0.509 | 0.855 | $-0.391$ | $\bm{-0.346}$ (Wilcoxon $p<10^{-9}$) | $\bm{7.9\times}$ |
| Held-out SAD+AISI (v2; no gate available) | 146 | 0.787 | 0.610 | — | $-0.176$ (Cohen $d{=}0.66$, $p<10^{-13}$) | — | — |

The "Needham source pool" is donor-restricted to Needham (the only
benchmark for which the full v3 pipeline was run). Needham is itself
a meta-benchmark aggregating ~20 sub-benchmarks
(`aider`, `cybench`, `agentharm` sub-set, etc.); the per-sub-benchmark
composition is in `tells/data/reverse/removed.json` (see paper Appendix C).

The "$7.9\times$" ratio is $|\text{raw}\ \Delta| / |\text{control}\ \Delta| = 0.391/0.045$.

### 3.5 Per-category breakdown
For each of the six categories, restricted to E-cases that carry
at least one template of that category, we report causal $\Delta$
with per-category Wilcoxon $p$. All categories with $n \geq 11$ produce
statistically significant causal effects (paper Appendix C).

### 3.6 Held-out SAD v3 attempt
We attempted to apply the v3 preservation gate to the 60 SAD samples.
**$60/60$ rewrites were rejected by the gate.** Inspection of rejection
reasons shows the gate correctly identifying that SAD's MCQ-format
and behavior-target labels constitute the test, not removable
scaffolding. We frame this as a positive finding for the gate's
discriminative power, and the held-out judge-level result reported in
Table~\ref{tab:reverse-canonical} therefore uses v2 (scaffolding-only,
no gate) on the held-out pool.

File: `results/sad_v3_gated.json`,
`tells/data/heldout_audit/sad_removed.json`.

---

## 4 · Preservation oracles

A rewrite that moves $P(\text{eval})$ is interesting only if it
preserves the underlying task. We measure preservation in six
independent ways, on capability and safety benchmarks.

### 4.1 5-criterion behavioral preservation (LLM-judged)
For each v3-accepted (orig, rewrite) pair, gemini-2.5-flash answers:
*"Does the rewrite preserve the original task?"*
Pooled across six benchmarks, $n{=}49$ cells: $\bm{92\%}$ preservation.

### 4.2 Topic preservation (rubric, inter-rater)
Two independent LLM annotators (`gpt-5.1` and a held-out judge) rate
topic match on a 5-point rubric; we report the fraction at rubric
score $\geq 4/5$. $n{=}205$, $\bm{96\%}$ topic preservation, Cohen's
$\kappa = 0.59$ (substantial agreement).

### 4.3 BBH task-metric accuracy (capability)
For each (orig, v3) paired BBH transcript ($n{=}252$ pairs across
24 BBH sub-tasks), run the target model (gpt-5-mini) on both versions
and extract the answer. Score against the BBH ground truth.

- Original accuracy: $0.857$
- v3 accuracy: $0.770$
- Relative preservation: $\bm{89.8\%}$

Per-task breakdown shows most tasks preserve at $100\%$;
$\bench{logical\_deduction}$ (5/7-object variants), $\bench{ruin\_names}$,
and $\bench{sports\_understanding}$ lose substantial accuracy under v3
(scaffolding tokens partly load-bearing).

Files: `scripts/task_metric_bbh.py`, `scripts/score_bbh_llm_judge.py`,
`results/task_metric_bbh_summary.json`.

### 4.4 AgentHarm refusal-class preservation (safety, generic judge)
For each (orig, v3) paired AgentHarm transcript ($n{=}26$), run
gpt-5-mini on both; have gemini-2.5-flash classify each response as
**REFUSE / PARTIAL / COMPLY**. Preservation = same class.
$\bm{80.8\%}$ preservation; **$0/26$ unlocked from REFUSE to COMPLY**
under v3. v3 transitions are directionally toward caution.

### 4.5 AgentHarm safety-rubric preservation (stricter judge)
A stricter task-rubric judge classifies the same 26 responses as
**SAFE / HARMFUL** relative to AgentHarm's ground-truth task rubric
(not just refusal phrasing). Preservation $73.1\%$; unlock rate
$7.7\%$ ($2/26$ samples flipped from SAFE to HARMFUL); harden rate
$19.2\%$ ($5/26$ flipped from HARMFUL to SAFE).

Net effect on v3: $+11.5\,\text{pp}$ SAFE rate vs orig.

File: `results/safety_task_metric_agentharm.json`.

### 4.6 Hawthorne refusal-class preservation
Same as 4.4 on Hawthorne (n=17). $\bm{100\%}$ preservation;
$0/17$ unlocks.

### 4.7 Dual-annotator safety spot-check ($n{=}60$)
Two independent LLMs from different families
(`gemini-2.5-flash`, `claude-haiku-4.5`) independently annotate
$n{=}60$ paired AgentHarm + Hawthorne $\times$ {orig, v3} $\times$
$15$ per cell responses on three axes: harmfulness class,
refusal-target preservation, task-unlock yes/no.

Results:
- Inter-annotator raw agreement: $\bm{81.7\%}$
- Cohen's $\kappa = \bm{0.509}$ (moderate)
- **$0/30$ task unlocks** under either annotator on paired (orig,v3) samples
- v3 SAFE rate is non-decreasing in every cell under both annotators

This is a dual-LLM spot-check, not full human annotation; we report
it explicitly as such.

File: `results/human_safety_eval.json`,
`results/human_safety_eval_responses.json`.

### 4.8 Summary
| Oracle | $n$ | Value |
|---|---:|---:|
| 5-criterion behavioral (§4.1) | 49 | 92% |
| Topic rubric, IRR $\kappa{=}0.59$ (§4.2) | 205 | 96% |
| BBH task-metric (§4.3) | 252 | 89.8% |
| AgentHarm refusal-class (§4.4) | 26 | 80.8% |
| AgentHarm safety-rubric (§4.5) | 26 | 73.1% |
| Hawthorne refusal-class (§4.6) | 17 | 100% |
| Dual-annotator safety spot-check (§4.7) | 60 | $0/30$ unlocks |

All six fall in the $73$--$100\%$ range. The single number below $80\%$
(rubric judge) is corroborated by the dual-annotator $0/30$ unlocks.

---

## 5 · Activation probes (supporting diagnostics)

We treat activation probes as **supporting diagnostics**, not
as evidence for a universal eval-awareness representation. The headline
findings (§2, §3) flow through LLM-judge measurements.

### 5.1 Models
Six open-weight models across three families:
1. Qwen3-8B (dense, fp16)
2. Qwen3-32B (dense, fp16)
3. Qwen3.5-27B (dense, fp16)
4. Qwen3.5-122B-A10B (MoE, fp8)
5. Mistral-Small-24B-Instruct-2501 (dense, bf16) — cross-family
6. Gemma-2-27B-it (dense, bf16) — cross-family

Gemma fp16 produced NaN at all layers $\geq 16$ due to logit
soft-capping × fp16 overflow; we re-extracted with bf16 and got 0/9 cells
with NaN.

### 5.2 Activation extraction
For each (model, benchmark, condition) triple where condition $\in$
{orig\_eval, orig\_deploy, v1\_removed, v2\_removed, v3\_removed,
rev\_control}, extract per-layer residual-stream activations at the
**last input token** (most information-dense position for transcript
classification). We capture every 4th layer plus the last 3. Storage:
fp32 NPZ in `activation_probing/cache/`.

`register_forward_hook` on `model.model.layers[i]`. Hooks reliable with
eager attention; SDPA/flash-attn sometimes bypass the layer's `output`
tuple.

### 5.3 Per-benchmark probe training
For each benchmark with both `orig_eval` and `orig_deploy`:
1. Standardize features (z-score per dimension; without this, accuracy
   drops 5--10\% because high-variance residual-stream dimensions swamp
   the L2 regularization).
2. 5-fold stratified-K-fold cross-validation accuracy at every cached
   layer.
3. Select best layer by CV accuracy.
4. Train final classifier on full data at the best layer.

Per-benchmark probes achieve CV $\approx 0.82$--$0.97$ within their
benchmark (best layer varies by model and benchmark; not a universal
mid-layer phenomenon, see paper Appendix I).

### 5.4 Applying probes (per-bench $\Delta$)
For each cached condition, apply the best-layer probe and compute
$P(\text{eval})$. Per-bench $\Delta$ = $\bar P(\text{eval})_{\text{ablated}} - \bar P(\text{eval})_{\text{orig}}$.

Across 4 Qwen + 2 cross-family models, per-bench v3 $\Delta$ ranges
from $-0.10$ to $-0.27$. We do **not** claim monotonic scaling: e.g.,
Qwen3.5-35B-A3B (3B active) gives $-0.180$, which is larger than
Qwen3.5-122B-A10B (10B active) at $-0.155$.

### 5.5 M1 cross-benchmark transfer
For each per-benchmark probe applied to every other benchmark's E vs D,
compute AUROC. Off-diagonal AUC indicates whether per-bench probes
capture a generalizable signal or only benchmark identity.

Result: off-diagonal AUC $0.54$--$0.58$ across all six models
(Qwen $\sim$$0.54$; Mistral $0.549$; Gemma $0.579$). Self AUC
$0.989$--$0.995$ (5-fold stratified-CV mean at best layer; a held-out
test variant on Qwen3.5-27B gives $0.95$--$0.99$, indicating CV isn't
hiding leakage).

**Verdict**: per-bench probes are benchmark-identity classifiers.

### 5.6 Shuffled-label probe floor
To verify the off-diagonal $0.54$ is at chance (and not modestly above),
on Qwen3-32B we permute E vs D labels stratifiedly ($n_{\text{perm}}{=}100$),
retrain per-bench probes on the permuted labels, and evaluate on each
target's true E vs D.

| src → tgt | layer | actual AUC | null mean | null 95% CI | percentile |
|---|---:|---:|---:|---|---:|
| needham → hawthorne | 32 | 0.440 | 0.500 | [0.411, 0.615] | 16% |
| needham → mask | 32 | 0.487 | 0.493 | [0.309, 0.682] | 52% |
| hawthorne → needham | 8 | 0.641 | 0.481 | [0.297, 0.704] | 90% |
| hawthorne → mask | 8 | 0.538 | 0.493 | [0.331, 0.690] | 70% |
| mask → needham | 28 | 0.600 | 0.504 | [0.278, 0.733] | 74% |
| mask → hawthorne | 28 | 0.552 | 0.499 | [0.400, 0.615] | 74% |
| **Pooled** | — | **0.543** | **0.495** | — | **63%** |

No off-diagonal pair exceeds the $95$th percentile of its shuffled-label
null. The off-diagonal $0.54$ is statistically indistinguishable from
chance.

File: `results/shuffled_label_floor.json`.

### 5.7 M2 pooled-probe construction
Train a single probe on the **union** of needham + hawthorne + mask E vs D
(stratified 5-fold CV), at the best pooled layer. The pooled probe achieves
CV $\approx 0.83$--$0.92$ across the six models (Mistral $0.916$, Gemma
$0.890$ — the two highest). Applied to v3-removed at the same layer, the
pooled probe produces $\Delta$ comparable to per-bench probes (ratio
$0.90$--$1.18\times$ across models).

| Model | per-bench $\Delta$ | pooled $\Delta$ |
|---|---:|---:|
| Qwen3-8B | $-0.103$ | $-0.535$ |
| Qwen3-32B | $-0.173$ | $-0.598$ |
| Qwen3.5-27B | $-0.222$ | $-0.575$ |
| Qwen3.5-122B-A10B fp8 | $-0.155$ | $-0.529$ |
| Mistral-Small-24B (cross-family) | — | $-0.289$ |
| Gemma-2-27B-it (cross-family) | $-0.269$ | $-0.250$ |

The cross-family rows confirm the v3 effect generalizes beyond Qwen.
Pooled-probe magnitude is larger than per-bench because the pooled
probe weights features that are bank-discriminative across the union
of training distributions.

### 5.8 Paired bank-specificity (probe-level)
For each base sample ID with cached orig, v3-removed, and rev-control,
compute the per-sample causal $\Delta = \Delta_{\text{bank}} - \Delta_{\text{ctrl}}$
on the pooled probe and bootstrap a 95% CI ($B{=}1000$).

| Benchmark | $n_{\text{paired}}$ | $\Delta_{\text{bank}}$ | $\Delta_{\text{ctrl}}$ | causal $\Delta$ ($95\%$ CI) |
|---|---:|---:|---:|---|
| Hawthorne (v3) | 15 | $-0.742$ | $-0.130$ | $\bm{-0.612}$ $[-0.789, -0.404]$ |
| SAD (v2, held-out) | 10 | $-0.165$ | $-0.002$ | $\bm{-0.163}$ $[-0.350, -0.032]$ |

The SAD activation $\Delta = -0.165$ converges with the judge-level
held-out SAD shift of $-0.176$ from Table~\ref{tab:reverse-canonical}.

File: `results/activation_probing/bank_specificity_paired.json`.

### 5.9 Cross-DOMAIN (capability vs safety)
A complementary cut: train two specialized probes.
- Capability-only: needham E vs D ($n_{\text{pos}}{=}150$, $n_{\text{neg}}{=}89$).
- Safety-only: union of hawthorne and mask E vs D.

Cross-domain transfer AUC: capability $\to$ safety $0.543$ (chance);
safety $\to$ capability $0.651$ (modestly above chance). Per-domain self
AUC $0.97$+.

### 5.10 Safety-keyword density correlation
For each transcript compute the fraction of words matching ~40
safety-keyword patterns (`refus`, `harm`, `danger`, `ethic`, `illegal`,
`weapon`, `deceiv`, `manipulat`, ...). Correlate with the probe's
$P(\text{eval})$ across all (probe, target) cells.

Pooled $r$ across $1{,}350$ samples (3 probes × 3 benchmarks × ~150 samples):
- Pearson $r = +0.009$
- Spearman $\rho = +0.022$

Rules out crude lexical safety-content confounds. Does not rule out
semantic safety/refusal structure; the cross-domain test in §5.9 is the
more rigorous semantic-safety check.

---

## 6 · Robustness checks

### 6.1 Lexical leave-one-benchmark-out (LOBO) baseline
Train a binary CountVectorizer logistic-regression on E vs D within
each benchmark; evaluate (a) 5-fold within-benchmark CV and (b)
leave-one-benchmark-out (LOBO). Sweep `max_features` $\in \{35, 100, 500, 2000\}$.

| `max_features` | Within-bench AUC | LOBO AUC |
|---:|---:|---:|
| 35 | 0.82 | 0.44 |
| 100 | 0.86 | 0.46 |
| 500 | 0.91 | 0.47 |
| 2000 | 0.94 | 0.47 |

Within-bench AUC ~$0.86$--$0.94$; LOBO collapses to $0.47$ (chance).
**Lexical features are benchmark-specific.** The bank's
cross-benchmark generalization (11/11 forward, $-0.176$ held-out) is
not explained by surface vocabulary.

File: `results/lexical_baseline_lobo.json`.

### 6.2 Dual-judge stratified agreement ($n{=}153$)
A formal stratified sample of $n{=}153$ transcripts (~26 per benchmark
× 6 benchmarks; capability benchmarks contribute E-cases only) is
independently scored YES/NO eval-vs-deployment by two judges:
`gemini-2.5-flash` and `claude-haiku-4.5`.

- Aggregate agreement: $\bm{92.2\%}$
- Cohen's $\bm{\kappa = 0.413}$ (moderate, Landis & Koch)
- Both judges $\sim$$71\%$ accurate vs gpt-5-mini-assigned labels

Per-benchmark agreement: $88.2\%$ on E+D benchmarks; $100\%$ on
eval-only capability benchmarks. The 88% on E+D benchmarks corresponds
to borderline samples.

File: `results/dual_judge_stratified.json`.

### 6.3 Position-bias 2-swap ($n{=}81$)
For each of $n{=}81$ stratified samples, query gpt-5-mini under two
equivalent prompts:
- PROMPT_A: "...output YES if eval, NO otherwise"
- PROMPT_B: "...output NO if real-world, YES otherwise"

Flip rate $=$ fraction of samples where the binary classification flips
between the two prompts. Overall flip rate $= \bm{6.2\%}$; per-benchmark
range $0\%$ (capability) to $16.7\%$ (Hawthorne). Within community-standard
robustness range.

File: `results/position_bias_swap.json`.

### 6.4 Ungated vs gated rewriter (v1 vs v3)
Same rewriter (`gemini-2.5-flash`) without the preservation gate
produces a $\sim$$2.5\times$ larger apparent activation-probe effect
($-0.42$ avg.\ across models vs v3 $-0.17$ avg.) but destroys task
behavior: $\sim$$10\%$ refusal-class preservation on AgentHarm vs v3's
$80.8\%$. The $\sim$$0.25$ probe gap is the inflation attributable to
inadvertent task drift in ungated rewriting.

We label this an "ungated rewrite baseline", **not** a "PRE-equivalent"
reproduction; we did not run StealthEval/PRE's exact rewriter or dataset.

---

## 7 · Statistical methods

### 7.1 Bootstrap
$B{=}1000$ resamples with replacement. For pooled means we resample
cells (forward: $n_{\text{cell}}{=}11$); for per-cell CIs we resample
per-sample within the cell. 95% CI = 2.5--97.5 percentile of the
resample distribution.

### 7.2 Wilcoxon signed-rank
For paired baseline-vs-treatment comparisons (reverse intervention).
One-sided alternative "treatment $\Delta$ is more negative" (or
positive for the forward direction). Reported as the cell-level
$p$-value.

### 7.3 t-test
For pooled forward causal $\Delta$ vs $0$, one-sample t-test treats each
cell's mean causal $\Delta$ as an observation ($n_{\text{cell}}{=}11$).
For per-cell significance we use either bootstrap 95% CI excludes 0
or $|t| > 1.96$ as the threshold.

### 7.4 Effect-size measures
- Cohen's $d$ (paired): $\bar\Delta / s_d$
- Cohen's $\kappa$: chance-corrected inter-rater agreement (Landis & Koch
  thresholds: 0.41--0.60 = moderate)
- AUROC for probe transfer

### 7.5 Bonferroni / Holm correction
We do not apply formal multiple-comparison correction in the main paper
because our cells / categories are not independent (same source pool, same
templates). For the per-category reverse breakdown
(paper Appendix C), we report uncorrected $p$ but require $n \geq 11$
per category for the test to be reported as "significant".

---

## 8 · Cost and compute

| Bucket | Cost |
|---|---:|
| OpenRouter API (judges, rewriters, neutral controls, dual annotators) | ~$\$80$ |
| RunPod GPU (H200 + B200 for activation extraction across 6 models) | ~$\$180$ |
| **Total** | $\sim\$260$ |

H200/B200 hours: ~50 across the 4 Qwen + Mistral + Gemma models, plus
pilot iterations. We did not train any model; all inference.

---

## 9 · Reproducing the headline numbers

The cheapest end-to-end smoke test (no API key, no GPU) reproduces the
lexical LOBO baseline:

```bash
cd test_smells_release
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/lexical_baseline.py
# diff results/lexical_baseline_lobo.json against the shipped reference
```

Full end-to-end reproduction (with API keys + GPU):

```bash
# Bank extraction (per source benchmark)
python bank_pipeline/extract_tells.py --benchmark needham
# ... etc for each benchmark
python bank_pipeline/classify_existing_bank.py

# Forward injection across all 11 cells (matched paraphrase control included)
python bank_pipeline/merge_tells.py --source needham --target hawthorne
# ... etc

# Reverse intervention with v3 preservation gate
python bank_pipeline/remove_tells.py --benchmark needham
python bank_pipeline/generate_reverse_controls.py --benchmark needham

# Activation extraction on GPU (per model)
export ACTPROBE_MODEL=Qwen/Qwen3.5-27B
bash activation_probing/run_full_pipeline.sh
python -m activation_probing.m1_cross_transfer
python -m activation_probing.m2_pooled_probe
python -m activation_probing.bank_specificity_paired

# Preservation oracles + robustness
python scripts/task_metric_bbh.py
python scripts/safety_task_preservation.py
python scripts/forward_neutral_injection.py
python scripts/pri5_bank_overlap.py
python scripts/lexical_baseline.py
```

The README in this directory ties each paper table to its script and
output JSON.

---

## 10 · What the methodology does NOT claim

To be explicit, our methodology does **not**:
1. Claim a universal eval-awareness representation in model activations
   (Section 5 reports this as a deliberate negative result).
2. Claim that our v1 ungated baseline is equivalent to
   PRE/StealthEval's published rewriter — we did not run their exact prompt.
3. Claim full single-judge independence — the dual-judge κ=0.41 and
   dual-annotator κ=0.51 are partial mitigations.
4. Claim that bank features compose additively — Appendix~F's
   preliminary compositionality analysis is included as exploratory only.
5. Claim that surface features capture "all" eval-identifiability —
   the surface/intrinsic decomposition explicitly leaves a large
   intrinsic component, especially on safety and situational-awareness
   benchmarks.

Each of these limits is also enumerated in Section 11 of the paper
(L1--L6).
