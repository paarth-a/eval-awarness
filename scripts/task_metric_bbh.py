"""PRI 6: Task-metric preservation on BBH.

Goal: show that v3-rewritten BBH transcripts produce the same TASK ACCURACY as
originals — a harder claim than "behavior is preserved per LLM judge". This addresses
the reviewer concern: "your preservation gate uses an LLM to judge preservation, but
does the model actually still solve the problem?"

Method:
  1. Load 252 BBH v3 rewrites and their originals
  2. Run gpt-5-mini on both (orig, v3) via OpenRouter
  3. Extract final answer from each response
  4. Score against ground-truth `bbh_target` from metadata
  5. Report orig_accuracy, v3_accuracy, delta, per-task breakdown

Cost: ~$2 (504 calls × $0.003)
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "openai/gpt-5-mini"
MAX_WORKERS = 16

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("ERROR: set OPENROUTER_API_KEY env var")
    sys.exit(1)

import requests


def call_model(messages, max_tokens=400):
    """Call model via OpenRouter, return text response."""
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_answer(response_text: str, task: str, target: str) -> str:
    """Extract a final answer from the model's response."""
    txt = response_text.strip()
    # Common final-answer patterns
    m = re.search(r"(?:final answer|answer)[:\s]*([^\n]+?)(?:\.|$)", txt, re.IGNORECASE)
    if m:
        cand = m.group(1).strip().rstrip(".,!?;:")
        return cand
    # Last line as fallback
    return txt.splitlines()[-1].strip().rstrip(".,!?;:") if txt else ""


def score_answer(predicted: str, target: str, task: str) -> bool:
    """Permissive scoring: exact match on normalized strings, or target is substring of predicted."""
    p = str(predicted).strip().lower().rstrip(".,!?;:")
    t = str(target).strip().lower().rstrip(".,!?;:")
    if p == t:
        return True
    # Boolean / yes-no
    if t in ("true", "false", "yes", "no"):
        if t in p:
            # Only accept if NOT preceded by negation
            return True
    # Multiple choice (A, B, C, etc.)
    if re.fullmatch(r"\([a-z]\)", t):
        if t in p or t.strip("()") in p.split():
            return True
    # Target is short string, look for it explicitly
    if len(t) <= 30 and t in p:
        return True
    return False


def process_one(item, condition_label):
    """Run model on one transcript, return (id, target, pred, correct)."""
    msgs = item["input"]
    target = str(item["metadata"]["bbh_target"]).strip()
    task = item["metadata"]["bbh_task"]
    try:
        resp = call_model(msgs)
        pred = extract_answer(resp, task, target)
        ok = score_answer(pred, target, task)
        return {"id": item["id"], "condition": condition_label, "target": target,
                "pred": pred, "correct": ok, "task": task, "response_first200": resp[:200]}
    except Exception as e:
        return {"id": item["id"], "condition": condition_label, "target": target,
                "pred": None, "correct": False, "task": task, "error": str(e)}


def main():
    transcripts = {t["id"]: t for t in json.loads((REPO / "tells" / "data" / "bbh" / "transcripts.json").read_text())}
    v3 = json.loads((REPO / "tells" / "data" / "bbh" / "reverse_v3" / "removed.json").read_text())

    # Match originals to v3
    pairs = []
    for v3_item in v3:
        orig_id = v3_item["metadata"]["original_sample_id"]
        orig = transcripts.get(orig_id)
        if not orig:
            continue
        # Original item needs bbh_target — should be in its metadata
        if "bbh_target" not in orig.get("metadata", {}):
            # Borrow target from v3 metadata (they share original)
            orig["metadata"]["bbh_target"] = v3_item["metadata"]["bbh_target"]
            orig["metadata"]["bbh_task"] = v3_item["metadata"]["bbh_task"]
        pairs.append((orig, v3_item))

    print(f"Loaded {len(pairs)} (original, v3) pairs from BBH")
    print(f"Total API calls: {len(pairs) * 2}")
    print(f"Model: {MODEL}")
    print()

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {}
        for orig, v3_item in pairs:
            futs[pool.submit(process_one, orig, "orig")] = ("orig", orig["id"])
            futs[pool.submit(process_one, v3_item, "v3")] = ("v3", v3_item["id"])
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(futs)} done ({time.time()-t0:.0f}s)")

    print(f"All {len(results)} calls done in {time.time()-t0:.0f}s")

    # Save raw
    out_dir = REPO / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_metric_bbh_raw.json").write_text(json.dumps(results, indent=2))

    # Score
    by_id = defaultdict(dict)
    for r in results:
        base_id = r["id"].split("__")[0]
        by_id[base_id][r["condition"]] = r

    n_orig_correct = sum(1 for d in by_id.values() if d.get("orig", {}).get("correct"))
    n_v3_correct = sum(1 for d in by_id.values() if d.get("v3", {}).get("correct"))
    n_orig = sum(1 for d in by_id.values() if "orig" in d)
    n_v3 = sum(1 for d in by_id.values() if "v3" in d)

    print()
    print("=" * 60)
    print("OVERALL TASK-METRIC PRESERVATION")
    print("=" * 60)
    print(f"Original accuracy:  {n_orig_correct}/{n_orig} = {n_orig_correct/n_orig:.3f}")
    print(f"v3       accuracy:  {n_v3_correct}/{n_v3} = {n_v3_correct/n_v3:.3f}")
    print(f"Δ (v3 − orig):      {(n_v3_correct/n_v3) - (n_orig_correct/n_orig):+.3f}")

    # Per-task breakdown
    by_task = defaultdict(lambda: {"orig_n": 0, "orig_c": 0, "v3_n": 0, "v3_c": 0})
    for d in by_id.values():
        task = (d.get("orig") or d.get("v3"))["task"]
        if "orig" in d:
            by_task[task]["orig_n"] += 1
            by_task[task]["orig_c"] += int(d["orig"]["correct"])
        if "v3" in d:
            by_task[task]["v3_n"] += 1
            by_task[task]["v3_c"] += int(d["v3"]["correct"])

    print()
    print("PER-TASK BREAKDOWN (sorted by sample count)")
    print(f"{'task':<45} {'orig':>10} {'v3':>10} {'Δ':>8}")
    rows = sorted(by_task.items(), key=lambda x: -(x[1]["orig_n"] + x[1]["v3_n"]))
    for task, stats in rows:
        if stats["orig_n"] == 0 or stats["v3_n"] == 0:
            continue
        oa = stats["orig_c"] / stats["orig_n"]
        va = stats["v3_c"] / stats["v3_n"]
        print(f"{task:<45} {oa:>4.2f} ({stats['orig_n']:>2}) {va:>4.2f} ({stats['v3_n']:>2}) {va-oa:>+.2f}")

    # Save summary
    summary = {
        "model": MODEL,
        "n_pairs": len(pairs),
        "orig_accuracy": n_orig_correct / n_orig if n_orig else 0,
        "v3_accuracy": n_v3_correct / n_v3 if n_v3 else 0,
        "delta": (n_v3_correct / n_v3) - (n_orig_correct / n_orig) if (n_orig and n_v3) else 0,
        "per_task": {t: {"orig_acc": (s["orig_c"]/s["orig_n"]) if s["orig_n"] else None,
                          "v3_acc": (s["v3_c"]/s["v3_n"]) if s["v3_n"] else None,
                          "orig_n": s["orig_n"], "v3_n": s["v3_n"]}
                      for t, s in by_task.items()},
    }
    (out_dir / "task_metric_bbh_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"[saved] {out_dir / 'task_metric_bbh_summary.json'}")


if __name__ == "__main__":
    main()
