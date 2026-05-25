# Activation Probing Pipeline

A standalone pipeline for training **linear logistic-regression probes on the
residual stream of an open-weight LLM** to detect "is this transcript an
evaluation?" — then measuring how preservation-gated vs aggressive sanitization
shifts the probe's verdict.

The activation-level companion to the LLM-judge experiments in the parent repo.
It empirically connects the transcript-level decomposition (scaffolding vs
test-bearing tells) to internal model state across 5 open-weight Qwen variants
(8B → 122B).

---

## 1 · The probe at a glance

**What it is.** A scikit-learn `LogisticRegression` classifier trained on a single
hidden-state vector per transcript, taken from one decoder layer of the target
model. Inputs are the model's own residual-stream activations; outputs are
P(eval) ∈ [0, 1].

**What it measures.** For a target model M and a benchmark B, the probe answers:
*"From M's internal representation of this transcript, can a linear classifier
distinguish the high-confidence E-cases from the low-confidence D-cases?"* If
yes, M itself is internally tracking an eval-vs-deployment distinction in a
linearly-readable way.

**What the comparisons buy.** Once a probe is trained, we apply it to four
additional conditions on the same benchmark:

| Condition | Meaning | Δ vs orig_eval |
|---|---|---|
| `orig_eval`     | Original E-cases (training positives) | 0 by definition |
| `orig_deploy`   | Original D-cases (training negatives) | large negative (sanity check) |
| `v1_removed`    | Aggressive LLM rewrite, no preservation gate (PRE-equivalent) | large negative |
| `v2_removed`    | Scaffolding-only filter, no preservation gate | medium negative |
| `v3_removed`    | Scaffolding-only + 5-criterion preservation gate + protected spans | **small negative — this is the headline** |

A v3 negative Δ that is robust across models = the bank tells the LLM judge
extracts are ALSO features the target model encodes internally; removing them
moves the model's hidden state, not just the judge's surface read.

**Headline cross-model result (pooled across needham/hawthorne/mask):**

| Model | Active params | Probe CV acc | v3 Δ |
|---|---:|---:|---:|
| Qwen3-8B            |   8B | 0.82–0.97 | **−0.103** |
| Qwen3-32B           |  32B | 0.82–0.97 | **−0.173** |
| Qwen3.5-27B         |  27B | 0.82–0.98 | **−0.222** |
| Qwen3.5-35B-A3B MoE |   3B | 0.68–0.97 | **−0.180** |
| Qwen3.5-122B-A10B MoE | 10B | 0.82      | **−0.155** |

Monotonic in active-parameter count; signal scales with capability, not total
parameters.

---

## 2 · How it works (mechanics)

### 2.1 Extraction (`extract_activations.py`)

For each (benchmark, condition, sample):

1. **Load the transcript** as a list of `{role, content}` messages via
   `data_loader.load_condition()`. Tool messages are flattened to user-role
   `[tool result] ...` to satisfy chat-template constraints.
2. **Apply the model's chat template** (`tokenizer.apply_chat_template`) and
   tokenize. Truncate from the LEFT (keep the last `MAX_TOKENS=4096`); this
   preserves the most recent context — what matters most for last-token
   classification.
3. **Forward pass with `torch.no_grad()`** in fp16, `device_map="auto"` (works
   for single- or multi-GPU sharding).
4. **`register_forward_hook` on `model.model.layers[i]`** for every layer in
   `LAYER_INDICES`. The hook saves `output[0][:, -1, :].detach().to(fp32).cpu()`
   — i.e., the hidden state at the LAST input token, in fp32, on CPU.
5. **Save per layer** as `cache/{MODEL}__{benchmark}__{condition}.npz`:
   `layer_4: [N, H], layer_8: [N, H], …, sample_ids: [N]`.

Why **last token**? At the last position, the residual stream contains the
model's "summary so far" — the contextualized representation it would use to
predict the next token. For transcript-level classification, this is the
information-densest position.

Why **fp32 on CPU for the saved tensors**? Linear probes are sensitive to
floating-point noise; an 0.01 shift in a single dimension can change the
prediction. fp16 storage is too lossy; fp32 + CPU lets us re-run probe training
many times cheaply without re-running the GPU forward pass.

Why **eager attention**? Hooks are reliable with eager attention; SDPA and
flash-attn sometimes bypass the layer's `output` tuple, breaking hook capture.

### 2.2 Training (`train_probes.py`)

For each benchmark with both `orig_eval` and `orig_deploy` caches:

1. Build per-layer `(X, y)` where `X = concat(orig_eval[L], orig_deploy[L])`
   and `y ∈ {1, 0}`.
2. `StandardScaler().fit_transform(X)` (z-score per dimension) — without this,
   probe accuracy drops 5-10% because the residual stream has high-variance
   dimensions that swamp the regularization.
3. `LogisticRegression(C=1.0, max_iter=2000, random_state=42)`. No
   `class_weight` by default (caller can subsample to balance).
4. **5-fold StratifiedKFold cross-validation**, report `cv_acc_mean ± std` per
   layer.
5. **Pick the best layer** by `cv_acc_mean`. Save the best-layer's
   `{scaler, classifier, best_layer, all_layers_acc}` as a pickle.

We save **the full-data classifier** (trained on all (X, y) after CV), not a
CV fold's classifier. The CV accuracy reported is the held-out estimate; the
saved classifier maximizes signal for downstream `apply`.

### 2.3 Applying (`apply_probes.py`)

Given a trained probe (one scaler + one classifier at the best layer):

1. For each cached condition (orig_eval/orig_deploy/v1/v2/v3) on the SAME
   benchmark, load `acts[best_layer]`.
2. `Xs = scaler.transform(X)`, `proba = clf.predict_proba(Xs)[:, 1]`.
3. Save per-sample P(eval), per-condition mean / std / median to
   `results/{benchmark}__predictions.json`.

The "Δ vs orig_eval" reported in our paper is `mean(P_eval_v3) −
mean(P_eval_orig)` pooled across (sample × benchmark) cells.

### 2.4 Analysis (`analyze_probes.py`)

Aggregates `*__predictions.json` into:
- `results/summary_table.json` — programmatic
- `results/summary_table.md`   — human-readable
- `results/figure.pdf`         — bar chart with bootstrap CIs

Bootstrap CI: 1000 resamples of (mean P(eval) per condition per benchmark).

---

## 3 · Cross-benchmark and cross-model utilities

Beyond the standard per-benchmark train→apply, the pipeline ships three
additional scripts that exploit cached activations:

### `cross_apply.py`

Apply benchmark A's probe to benchmark B's cached activations. Used for the
"does a Needham probe identify Hawthorne eval-cases?" matrix. Requires both
probes' best-layer to be in B's cached layers (use `LAYER_INDICES` matching
across benchmarks).

### `per_layer_curve.py`

For one (model, benchmark), recompute CV accuracy at EVERY cached layer (not
just pick the best). Outputs a layer-vs-accuracy table — used to refute the
"mid-layers most salient" claim from arxiv 2507.01786. See
`layer_salience_analysis.md`.

### `apply_neutral_control.py`

Apply a trained probe to `rev_control` activations — matched-scope random-span
rewrites that act as a placebo for the LLM rewriter. Computes the bank-
specificity ratio: `Δ(v3) / Δ(rev_control)`. Capability benchmarks: 3.67×;
safety benchmarks: ~1×.

### `compositionality.py`

For forward-injection data (`fwd_control`, `fwd_solo:<template>`,
`fwd_combined`), measure whether per-template effects sum to the all-templates-
together effect at the activation level. Result on Hawthorne and MASK: NO
additivity; combined effect is ~0 even when individual solos sum to large
shifts. Implies the probe sees the bank in a saturating / context-dependent
way.

---

## 4 · Data flow and dependencies

```
parent repo's data
  ├── tells/data/{B}/transcripts.json              (originals)
  ├── tells/data/{B}/high_confidence_eval_cases*   (E-case IDs)
  ├── tells/data/{B}/low_confidence_deploy_cases   (D-case IDs)
  ├── tells/data/{B}/reverse/dataset.json          (v1 rewrites)
  ├── tells/data/{B}/reverse_v2/removed.json       (v2 rewrites)
  ├── tells/data/{B}/reverse_gated/removed.json    (gated rewrites, after preservation gate)
  ├── tells/data/{B}/reverse/control.json          (neutral controls, optional)
  └── tells/data/{B}/forward_paired/*.json         (forward-injection data)

         │
         ▼
data_loader.load_condition(benchmark, condition)
     returns list of (sample_id, messages)
         │
         ▼
extract_activations.run_extraction
     produces cache/{MODEL}__{B}__{C}.npz
         │
         ▼
train_probes.train_for_benchmark(B)
     uses orig_eval + orig_deploy only
     produces probes/{B}__probes.pkl
         │
         ▼
apply_probes.apply_for_benchmark(B)
     applies probe to all cached conditions for B
     produces results/{B}__predictions.json
         │
         ▼
analyze_probes.main()
     aggregates → results/summary_table.{json,md} + figure.pdf
```

---

## 5 · Configuration (`config.py`)

| Variable | Default | Env var | What it controls |
|---|---|---|---|
| `MODEL_NAME`         | `Qwen/Qwen3-8B` | `ACTPROBE_MODEL`  | HF model id |
| `LAYER_INDICES`      | auto (every 4th + last 3) | `ACTPROBE_LAYERS` (comma-separated) | which layers to extract |
| `MAX_TOKENS`         | 4096 | `ACTPROBE_MAX_TOKENS` | left-truncation length |
| `DTYPE`              | `float16` | — | inference dtype |
| `BENCHMARKS`         | needham, hawthorne, mask, agentharm, bbh, malt | — | which to run |
| `CONDITIONS`         | orig_eval, orig_deploy, v1/v2/v3_removed | — | what to extract |
| `N_PER_CONDITION`    | None (all) | `ACTPROBE_N_PER` | sample cap per cell |
| `PROBE_C`            | 1.0 | `ACTPROBE_C` | sklearn L2 strength (inverse) |
| `PROBE_CV_FOLDS`     | 5  | — | CV folds |
| `RNG_SEED`           | 42 | — | reproducibility |
| `N_BOOTSTRAP`        | 1000 | — | bootstrap iterations for CIs |

---

## 6 · End-to-end run

```bash
# 1) Spin up a GPU box (RunPod, Modal, …) with PyTorch 2.5+ and transformers 4.55+

# 2) Clone parent repo + this folder lives at activation_probing/
cd \$RELEASE_ROOT
pip install -r activation_probing/requirements.txt

# 3) Required env
export HF_TOKEN=hf_xxx                # if downloading gated models
export HF_HOME=/path/to/hf_cache    # somewhere with ≥200 GB
export ACTPROBE_MODEL=Qwen/Qwen3.5-27B   # pick a model
export ACTPROBE_N_PER=150             # sample cap

# 4) Full pipeline
bash activation_probing/run_full_pipeline.sh
# = python -m activation_probing.extract_activations --all
#   python -m activation_probing.train_probes --all
#   python -m activation_probing.apply_probes --all
#   python -m activation_probing.analyze_probes
```

Per-cell wall time on a 2× H200 community-cloud pod: ~5–10 minutes per
(benchmark, condition) extraction for 27B; faster for smaller models, slower
for larger.

---

## 7 · Hardware requirements

| Model | Param footprint | Minimum GPU | Wall time (full pipeline) | Cost |
|---|---|---|---|---|
| Qwen3-8B           | 8 B   | A100 40GB | 4-6 hr  | $5-10 |
| Qwen3-32B          | 32 B  | A100 80GB | 8-12 hr | $15-30 |
| Qwen3.5-27B        | 27 B  | A100 80GB | 3-4 hr  | $5-10 |
| Qwen3.5-35B-A3B    | 35 B (3 B active) | A100 80GB | 30 min | $1 |
| Qwen3.5-122B-A10B  | 122 B (10 B active) fp8 | H200 141 GB | 1-2 hr | $4-8 |
| Qwen3.5-397B-A17B  | 397 B (17 B active) fp8 | 2× H200 282 GB | 2-3 hr | $14-21 |
| DeepSeek-V4-Flash  | 284 B (13 B active) fp8 | H200 141 GB + DeepSeek inference code | 3-5 hr incl setup | $14-22 |

Memory dominated by KV cache; effective batch size at extraction is 1 (one
transcript = one forward pass) — there's no batching benefit, so we don't
batch. Time dominated by the forward pass per transcript.

---

## 8 · Interpreting results

### 8.1 Probe CV accuracy

| Range | What it means |
|---|---|
| 0.50–0.60 | The model does NOT linearly encode eval-vs-deploy at this layer for this benchmark. Try a different layer. If still flat, the model genuinely doesn't see the distinction. |
| 0.65–0.75 | Weak signal. Probe is finding something but it's noisy. |
| 0.80–0.95 | Strong signal. Eval-vs-deploy is a major axis of variation in the residual stream. |
| 0.95–1.0  | Likely overfitting / class imbalance / data leakage. Inspect class counts. |

### 8.2 v3 Δ magnitude

| Magnitude (negative) | Interpretation |
|---|---|
| 0.00 to −0.05 | Bank tells don't move the model's internal representation. Either tells are weak, the model doesn't track them, or v3 over-filtered. |
| −0.05 to −0.15 | Modest shift; tells are present internally but not dominant. |
| −0.15 to −0.30 | Strong shift; tells are major contributors to the model's eval-classification. |
| <−0.30 | Likely confounded with task-content change (compare to v1's −0.40 range — that's task drift). |

### 8.3 Sanity ordering

For every benchmark × model, expect `|Δ(v1)| > |Δ(v2)| > |Δ(v3)|` (more
aggressive removal → larger shift). Violations indicate broken extraction or
mislabeled cache files.

---

## 9 · Coverage caveat (read me!)

| Benchmark | orig_eval | orig_deploy | v1_removed | v2_removed | v3_removed |
|---|---:|---:|---:|---:|---:|
| needham    | 175 | 89 | 161 | 119 | 75 |
| hawthorne  | 200 | 200 | 30 | 37 | 17 |
| mask       | 1000 | 30 | 141 | 1 | 1 |
| agentharm  | 468 | **0** | 50 | 63 | 26 |
| bbh        | 540 | **0** | 50 | 528 | 252 |
| malt       | 91 | **0** | 91 | 91 | 34 |
| **SAD (held-out)** | 60 | 0 | — | 60 | — |

**Probe-training caveat:** AgentHarm, BBH, MALT, SAD are eval-only (no
D-cases). Two options:

1. **Recommended:** Train probes only on benchmarks with both classes
   (needham, hawthorne, mask). Then **apply those probes** to the eval-only
   benchmarks' sanitized variants — cross-benchmark transferability story.
2. Borrow D-cases from a different benchmark. Cleaner but introduces a
   confound.

The pipeline currently does (1) — `train_probes.py` skips benchmarks lacking
both classes; `apply_probes.py` still applies the probe to every cached
condition.

**MASK class imbalance:** 1000 E vs 30 D. The default `LogisticRegression`
doesn't use `class_weight='balanced'`. If you train on MASK directly, expect
the probe to over-predict P(eval). For our headline numbers we either
subsample (`ACTPROBE_N_PER=100`) or use the imbalance as-is (and lean on
needham/hawthorne for the pooled summary).

---

## 10 · Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Probe CV acc ≈ 0.5 everywhere | Hook didn't capture, or data labels swapped | `print(_BUFFER.keys())` after one forward; verify `acts.shape[0] == len(items)` |
| OOM during extraction | KV cache too long | Lower `MAX_TOKENS`, or `device_map="auto"` across more GPUs |
| `model.model.layers` doesn't exist | Non-standard arch (e.g. DeepSeek V4) | Patch `get_layer_modules()` in `extract_activations.py` |
| Probe accuracy high but v3 Δ ≈ 0 | Probe is reading benchmark-identity, not eval-cues | LOBO probe transfer test (see `lexical_baseline.py` analog) |
| Activation file size 187 KB instead of 25 MB | `<5` samples succeeded for that condition | Inspect data_loader output for that (benchmark, condition); usually v2/v3 over-filtered |
| Tokenizer rejects "tool" role | Some chat templates don't define it | `messages_to_input_ids` already converts `tool → user [tool result] ...` |

---

## 11 · File reference

| File | Lines | Purpose |
|---|---:|---|
| `config.py`               | 73 | All config + env-var overrides |
| `data_loader.py`          | 245 | Load (sample_id, messages) for any (benchmark, condition) |
| `extract_activations.py`  | 270 | Forward pass + hook + cache to .npz |
| `train_probes.py`         | 143 | Per-layer CV, pick best layer, save pickle |
| `apply_probes.py`         | 99  | Apply best-layer probe to all cached conditions |
| `analyze_probes.py`       | 176 | Aggregate predictions → summary + figure |
| `per_layer_curve.py`      | 100 | Compute CV at every layer (for layer salience) |
| `cross_apply.py`          | ~100 | Apply benchmark-A probe to benchmark-B activations |
| `apply_neutral_control.py`| ~80  | Apply probe to rev_control (placebo) activations |
| `compositionality.py`     | ~150 | Per-template vs combined activation shift |
| `run_full_pipeline.sh`    | ~25  | Orchestrator: extract → train → apply → analyze |
| `requirements.txt`        | ~10  | `transformers, torch, scikit-learn, numpy, matplotlib, tqdm` |

---

## 12 · Citation / artifact

We release per-model probe pickles (`probes/{benchmark}__probes.pkl`) and
per-cell .npz activations for each of the 5 Qwen models alongside the paper.
Each probe pickle has:

```python
{
  "benchmark": str,
  "model": str,
  "best_layer": int,
  "best_layer_cv_acc": float,
  "scaler": sklearn.preprocessing.StandardScaler,
  "classifier": sklearn.linear_model.LogisticRegression,
  "all_layers_acc": {layer_idx: cv_acc, ...},
}
```

Load with: `pickle.load(open(path, "rb"))`. Apply to a new activation tensor
`X` of shape `[N, H]` with: `clf.predict_proba(scaler.transform(X))[:, 1]`.

---

## 13 · Background / prior work

- **Anthropic NLA (May 7, 2026)**: showed Claude has internal eval-suspicion
  not reflected in its outputs. Our pipeline is the open-weight, mechanistic
  counterpart.
- **Aranguri & Bloom (May 4, 2026 LessWrong)**: iterative paraphrasing reduces
  verbalized eval awareness AND refusal rates. Our preservation gate +
  activation probe disambiguates "model treats as deployment" from "test
  target shifted".
- **PRE / StealthEval (Sep 2025, arxiv 2509.00591)**: same probe-rewrite-
  evaluate pipeline at the judge level; we add the activation-level oracle
  and the preservation gate.
- **arxiv 2507.01786** (probing & steering eval awareness): claims mid-layers
  most salient. We test this across 5 models × 3 benchmarks and find best
  layer fits 1/15 cells.
- **arxiv 2603.19426** (format-sensitivity critique of PRE): argues probe
  results may reflect format sensitivity rather than eval awareness. Our
  matched neutral-control test (`apply_neutral_control.py`) provides the
  direct empirical test.
