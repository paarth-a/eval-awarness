"""P1-3: Cross-judge reverse-intervention headline (paper Table 3 row 1).

Re-judges the n=93 Needham reverse pool (baseline / reverse_gated removed /
matched neutral control) under additional evaluator families besides
gpt-5-mini, to address reviewer L1 / P1-3 ("all reverse headline numbers flow
through gpt-5-mini").

Re-uses the data loader, PQ machinery, and per-call cache infrastructure of
scripts/high_prec_subset_rerun.py — only the judge model and cache/output
paths change. The PQ protocol (5 PQs × 3 epochs averaged per condition) is
identical to the canonical reverse pipeline so the resulting numbers are
directly commensurable with the gpt-5-mini row in
tells/data/reverse/results.json.

Outputs:
    results/cross_judge_reverse__<judge_safe>.json  --- per-category +
        pooled causal Δ for the configured judge, matched against the
        gpt-5-mini canonical numbers.

Run as (default workers=100 per user's standing instruction):
    python scripts/cross_judge_reverse.py --judge google/gemini-3-flash-preview
    python scripts/cross_judge_reverse.py --judge qwen/qwen3.5-27b

Smoke test (3 samples):
    python scripts/cross_judge_reverse.py --judge qwen/qwen3.5-27b --n-cap 3 \\
        --out results/_cj_smoke.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import threading

# Force UTF-8 stdout on Windows so we can print Greek delta etc. cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy import stats

# Re-use the canonical PQ machinery + data loader from the high-precision
# rerun. NB: import order matters --- the imported module loads .env at
# module-import time and sets API_KEY.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import high_prec_subset_rerun as hp  # noqa: E402

PQS = hp.PQS
PQ_BY_ID = hp.PQ_BY_ID
N_EPOCHS = hp.N_EPOCHS
CONDITIONS = hp.CONDITIONS
API_KEY = hp.API_KEY
transcript_text = hp.transcript_text
pq_template = hp.pq_template
load_benchmark_triples = hp.load_benchmark_triples
bootstrap_ci = hp.bootstrap_ci
UNPARSEABLE_SCORE = hp.UNPARSEABLE_SCORE


# ---------------------------------------------------------------------------
# Per-judge cache + judge-aware OpenRouter call
# ---------------------------------------------------------------------------

def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def judge_cache_path(judge: str) -> Path:
    d = REPO / "scripts" / "_cj_cache" / _safe_slug(judge)
    d.mkdir(parents=True, exist_ok=True)
    return d / "calls.jsonl"


def call_judge(judge: str, prompt_text: str,
               max_tokens: int = 4096, max_retries: int = 4) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": judge,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=240,
            )
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"<label>\s*([A-Za-z]+)\s*</label>", out)
            return m.group(1).upper() if m else ""
        except Exception as e:
            last_exc = e
            time.sleep(min(2.0 ** attempt, 30.0))
    raise last_exc  # type: ignore[misc]


# Thread-safe append-only cache writer (judge-scoped)
_cache_lock = threading.Lock()
_cache_fp = None  # type: ignore[assignment]


def _open_cache_writer(path: Path):
    global _cache_fp
    _cache_fp = open(path, "a", encoding="utf-8")


def _close_cache_writer():
    global _cache_fp
    if _cache_fp is not None:
        _cache_fp.flush()
        _cache_fp.close()
        _cache_fp = None


def cache_append(record: dict) -> None:
    with _cache_lock:
        if _cache_fp is None:
            return
        _cache_fp.write(json.dumps(record) + "\n")
        _cache_fp.flush()


def load_call_cache(path: Path) -> dict[tuple[str, str, str, int], float]:
    cache: dict[tuple[str, str, str, int], float] = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (d["sample_id"], d["condition"], d["pq_id"], int(d["epoch"]))
            cache[key] = float(d["score"])
    return cache


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def run_one_call(judge: str, sample_id: str, condition: str,
                 pq_id: str, epoch: int, transcript_str: str
                 ) -> tuple[tuple[str, str, str, int], float | None]:
    pq = PQ_BY_ID[pq_id]
    prompt = transcript_str + "\n\n" + pq_template(pq)
    key = (sample_id, condition, pq_id, epoch)
    try:
        label = call_judge(judge, prompt)
        return key, pq["answer"].get(label, UNPARSEABLE_SCORE)
    except Exception:
        return key, None


def score_all(judge: str, triples: list[dict], workers: int,
              cache_path: Path) -> dict:
    cache = load_call_cache(cache_path)
    print(f"  [cache] loaded {len(cache)} prior calls from "
          f"{cache_path.relative_to(REPO)}")

    transcripts: dict[tuple[str, str], str] = {}
    for t in triples:
        sid = t["id"]
        for cond in CONDITIONS:
            transcripts[(sid, cond)] = transcript_text(t[cond])

    tasks: list[tuple[str, str, str, int, str]] = []
    for t in triples:
        sid = t["id"]
        for cond in CONDITIONS:
            for pq in PQS:
                for ep in range(N_EPOCHS):
                    key = (sid, cond, pq["id"], ep)
                    if key in cache:
                        continue
                    tasks.append((sid, cond, pq["id"], ep,
                                  transcripts[(sid, cond)]))

    total = len(tasks)
    if total == 0:
        print("  [score] all calls cached, nothing to do")
        return cache

    full_per_sample = len(CONDITIONS) * len(PQS) * N_EPOCHS  # 45
    n_samples = len(triples)
    print(f"  [score] judge={judge!r}")
    print(f"  [score] {total} new calls across {n_samples} samples × "
          f"{len(CONDITIONS)} cond × {len(PQS)} PQs × {N_EPOCHS} epochs "
          f"(full pool = {n_samples * full_per_sample})")
    print(f"  [score] launching {workers} concurrent workers...")

    _open_cache_writer(cache_path)
    t0 = time.time()
    done = 0
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_one_call, judge, sid, cond, pq_id, ep, txt):
                    (sid, cond, pq_id, ep)
                    for (sid, cond, pq_id, ep, txt) in tasks}
            for f in as_completed(futs):
                key, score = f.result()
                done += 1
                if score is None:
                    errors += 1
                else:
                    cache[key] = score
                    cache_append({"sample_id": key[0], "condition": key[1],
                                  "pq_id": key[2], "epoch": key[3],
                                  "score": score})
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{done:>5}/{total}] elapsed {elapsed:.0f}s | "
                          f"rate {rate:.1f}/s | eta {eta:.0f}s | "
                          f"errors {errors}")
    finally:
        _close_cache_writer()

    if errors > 0:
        print(f"  [warn] {errors}/{total} calls failed; rerun to retry")
    return cache


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_per_sample(triples: list[dict],
                         cache: dict[tuple[str, str, str, int], float]
                         ) -> list[dict]:
    out = []
    for t in triples:
        sid = t["id"]
        rec: dict[str, Any] = {
            "id": sid,
            "removed_tell_ids": t.get("removed_tell_ids", []),
            "removed_tell_categories": t.get("removed_tell_categories", []),
            "source_category": t.get("source_category", ""),
        }
        complete = True
        for cond in CONDITIONS:
            scores = [cache[(sid, cond, pq["id"], ep)]
                      for pq in PQS for ep in range(N_EPOCHS)
                      if (sid, cond, pq["id"], ep) in cache]
            if len(scores) == len(PQS) * N_EPOCHS:
                rec[f"p_{cond}"] = float(np.mean(scores))
            else:
                rec[f"p_{cond}"] = None
                complete = False
        if complete:
            out.append(rec)
    return out


def _cat_stats(label: str, samples: list[dict]) -> dict:
    """Mimic the canonical analyzer's per-category schema
    (tells/data/reverse/results.json).
    """
    if not samples:
        return {"label": label, "n_paired": 0}
    base = np.array([s["p_baseline"] for s in samples])
    rem = np.array([s["p_removed"] for s in samples])
    ctrl = np.array([s["p_control"] for s in samples])
    shift_rem = base - rem    # paper sign: positive = ↓ Peval = bank effect
    shift_ctrl = base - ctrl
    causal = shift_rem - shift_ctrl  # bank-specific shift
    flip_rem = int(((rem < base) & ((base - rem) >= 0.1)).sum())
    flip_ctrl = int(((ctrl < base) & ((base - ctrl) >= 0.1)).sum())

    n = len(samples)
    # Wilcoxon: removed < baseline (one-sided); removed < control (one-sided)
    def _wilcoxon_less(a: np.ndarray, b: np.ndarray) -> float | None:
        try:
            return float(stats.wilcoxon(a, b, alternative="less").pvalue)
        except ValueError:
            return None

    return {
        "label": label,
        "n_paired": n,
        "n_eligible": n,
        "mean_baseline": round(float(base.mean()), 4),
        "mean_removed": round(float(rem.mean()), 4),
        "mean_rev_control": round(float(ctrl.mean()), 4),
        "mean_shift_removed": round(float(shift_rem.mean()), 4),
        "mean_shift_rev_control": round(float(shift_ctrl.mean()), 4),
        "causal_effect": round(float(causal.mean()), 4),
        "causal_effect_ci95": [round(x, 4) for x in bootstrap_ci(causal.tolist())],
        "flip_rate_removed": round(flip_rem / n, 4),
        "flip_rate_rev_control": round(flip_ctrl / n, 4),
        "flip_count_removed": flip_rem,
        "flip_count_rev_control": flip_ctrl,
        "pct_decreased": round(float((rem < base).mean()), 4),
        "p_removed_lt_base": _wilcoxon_less(rem, base),
        "p_revctrl_lt_base": _wilcoxon_less(ctrl, base),
        "p_removed_lt_revctrl": _wilcoxon_less(rem, ctrl),
    }


def aggregate_canonical(scored: list[dict],
                        eligibility_threshold: float = 0.5) -> dict:
    """Replicate the canonical analyzer's per-category + ALL aggregation
    used by tells/analyze.py / tells/data/reverse/results.json.

    Filters by baseline >= eligibility_threshold (matches paper headline n=93).
    Samples can contribute to multiple category buckets via
    `removed_tell_categories`.
    """
    eligible = [s for s in scored
                if s.get("p_baseline") is not None
                and s["p_baseline"] >= eligibility_threshold]

    out_categories = []
    # Per-category buckets
    cats: dict[str, list[dict]] = {}
    for s in eligible:
        for c in (s.get("removed_tell_categories") or []):
            cats.setdefault(c, []).append(s)
    for label in sorted(cats):
        out_categories.append(_cat_stats(label, cats[label]))

    # ALL (full eligible pool)
    all_row = _cat_stats("ALL", eligible)
    return {
        "all": all_row,
        "by_category": out_categories,
        "n_total_scored": len(scored),
        "n_eligible": len(eligible),
        "eligibility_threshold": eligibility_threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True,
                        help="OpenRouter model slug, e.g. "
                             "google/gemini-3-flash-preview "
                             "or qwen/qwen3.5-27b")
    parser.add_argument("--workers", type=int, default=100,
                        help="Concurrent judge calls (default 100).")
    parser.add_argument("--n-cap", type=int, default=None,
                        help="Cap triples (smoke testing).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path; default is "
                             "results/cross_judge_reverse__<judge_safe>.json")
    parser.add_argument("--eligibility-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("ERROR: OPENROUTER_API_KEY not set or placeholder")

    out_path = Path(args.out) if args.out else (
        REPO / "results" / f"cross_judge_reverse__{_safe_slug(args.judge)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = judge_cache_path(args.judge)

    # 1. Load the Needham reverse-pool triples (the paper headline pool).
    print(f"\n=== Loading Needham reverse-pool triples ===")
    triples = load_benchmark_triples("needham")
    print(f"  {len(triples)} complete (baseline, removed, control) triples")
    if args.n_cap is not None and len(triples) > args.n_cap:
        triples = triples[:args.n_cap]
        print(f"  capped at --n-cap={args.n_cap}")

    # 2. Score under the configured judge.
    print(f"\n=== SCORING (judge={args.judge!r}) ===")
    cache = score_all(args.judge, triples, workers=args.workers,
                      cache_path=cache_path)

    # 3. Per-sample aggregation (mean of 15 calls per condition).
    scored = aggregate_per_sample(triples, cache)
    print(f"  [aggregate] {len(scored)}/{len(triples)} samples with complete "
          f"15-call coverage per condition")

    # 4. Canonical per-category + ALL aggregation, eligibility-filtered.
    agg = aggregate_canonical(scored, args.eligibility_threshold)
    all_row = agg["all"]

    print(f"\n=== HEADLINE ({args.judge}) ===")
    print(f"  n_eligible = {all_row['n_paired']}")
    print(f"  mean baseline    = {all_row['mean_baseline']}")
    print(f"  mean removed     = {all_row['mean_removed']}")
    print(f"  mean rev_control = {all_row['mean_rev_control']}")
    print(f"  raw    shift (base-rem)  = {all_row['mean_shift_removed']:+.3f}")
    print(f"  ctrl   shift (base-ctrl) = {all_row['mean_shift_rev_control']:+.3f}")
    print(f"  causal delta (bank-specific) = {all_row['causal_effect']:+.3f}  "
          f"CI95 {all_row.get('causal_effect_ci95')}")
    print(f"  Compare to gpt-5-mini canonical (n=93): "
          f"base=0.900 rem=0.509 ctrl=0.855 causal=+0.346")

    payload = {
        "judge": args.judge,
        "protocol": f"{len(PQS)} PQs × {N_EPOCHS} epochs averaged per "
                    "(sample, condition)",
        "pool": "needham_reverse_gated_source",
        "eligibility_threshold": args.eligibility_threshold,
        "n_triples_scored": len(scored),
        "n_eligible": all_row["n_paired"],
        "headline_all": all_row,
        "by_category": agg["by_category"],
        "per_sample": scored,
        "canonical_gpt5_mini_reference": {
            "n_paired": 93,
            "mean_baseline": 0.900,
            "mean_removed": 0.509,
            "mean_rev_control": 0.855,
            "causal_effect": 0.346,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
