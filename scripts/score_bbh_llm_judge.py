"""LLM-as-judge scorer for BBH. Robust to format variations (E vs (E), prose vs MCQ).

Cost: ~504 judge calls × $0.001 ≈ $0.50
"""
from __future__ import annotations
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
JUDGE = "google/gemini-2.5-flash"
API_KEY = os.environ.get("OPENROUTER_API_KEY")


JUDGE_PROMPT = """You are evaluating whether a model response correctly answers a BBH question.

Question / task:
{question}

Target answer: {target}
{content_hint}

Model response:
{response}

Did the model give the correct answer? The answer counts as correct if the response unambiguously identifies the target answer, even if formatted differently (e.g. "(E)" vs "E" vs the option content; "yes" vs "Yes — the sentence is plausible").

Reply with ONLY a single word: YES or NO."""


def call_judge(question, target, content_hint, response):
    prompt = JUDGE_PROMPT.format(
        question=question[:2000],
        target=target,
        content_hint=f"Target option content: \"{content_hint}\"" if content_hint else "",
        response=response[:1500],
    )
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": JUDGE, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 16, "temperature": 0.0},
        timeout=60,
    )
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"] or ""
    return out.strip().upper().startswith("Y")


def main():
    raw = json.loads((REPO / "results" / "task_metric_bbh_raw.json").read_text())
    transcripts = {t["id"]: t for t in json.loads((REPO / "tells" / "data" / "bbh" / "transcripts.json").read_text())}
    v3_items = {x["id"]: x for x in json.loads((REPO / "tells" / "data" / "bbh" / "reverse_gated" / "removed.json").read_text())}

    def judge_one(r):
        base = r["id"].split("__")[0]
        if r["condition"] == "orig":
            src = transcripts.get(base)
        else:
            src = v3_items.get(r["id"])
        if src is None:
            return r
        question = " ".join(m["content"] for m in src["input"])
        response = r.get("response_full") or r.get("response_first200") or ""
        if not response:
            r["correct_judge"] = False
            return r
        try:
            r["correct_judge"] = call_judge(question, r["target"], r.get("content_target"), response)
        except Exception as e:
            r["judge_err"] = str(e)
            r["correct_judge"] = False
        return r

    print(f"Judging {len(raw)} responses with {JUDGE}...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(judge_one, r) for r in raw]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(raw)} judged ({time.time()-t0:.0f}s)")

    (REPO / "results" / "task_metric_bbh_raw.json").write_text(json.dumps(raw, indent=2))

    by_id = defaultdict(dict)
    for r in raw:
        base = r["id"].split("__")[0]
        by_id[base][r["condition"]] = r

    n_orig_c = sum(1 for d in by_id.values() if d.get("orig", {}).get("correct_judge"))
    n_v3_c = sum(1 for d in by_id.values() if d.get("v3", {}).get("correct_judge"))
    n_orig = sum(1 for d in by_id.values() if "orig" in d)
    n_v3 = sum(1 for d in by_id.values() if "v3" in d)

    print()
    print("=" * 60)
    print("OVERALL TASK-METRIC PRESERVATION (LLM-as-judge scoring)")
    print("=" * 60)
    print(f"Original accuracy:  {n_orig_c}/{n_orig} = {n_orig_c/n_orig:.3f}")
    print(f"v3       accuracy:  {n_v3_c}/{n_v3} = {n_v3_c/n_v3:.3f}")
    print(f"Δ (v3 − orig):      {(n_v3_c/n_v3) - (n_orig_c/n_orig):+.3f}")
    print(f"Relative preservation: {n_v3_c/n_orig_c:.3f}")

    by_task = defaultdict(lambda: {"orig_n": 0, "orig_c": 0, "v3_n": 0, "v3_c": 0})
    for d in by_id.values():
        task = (d.get("orig") or d.get("v3"))["task"]
        if "orig" in d:
            by_task[task]["orig_n"] += 1
            by_task[task]["orig_c"] += int(d["orig"].get("correct_judge", False))
        if "v3" in d:
            by_task[task]["v3_n"] += 1
            by_task[task]["v3_c"] += int(d["v3"].get("correct_judge", False))

    print()
    print("PER-TASK (LLM-judge)")
    print(f"{'task':<45} {'orig':>12} {'v3':>12} {'Δ':>8}")
    rows = sorted(by_task.items(), key=lambda x: -(x[1]["orig_n"] + x[1]["v3_n"]))
    for task, stats in rows:
        if stats["orig_n"] == 0 or stats["v3_n"] == 0:
            continue
        oa = stats["orig_c"] / stats["orig_n"]
        va = stats["v3_c"] / stats["v3_n"]
        print(f"{task:<45} {oa:>5.2f} ({stats['orig_n']:>2}) {va:>5.2f} ({stats['v3_n']:>2}) {va-oa:>+.2f}")

    summary = {
        "model": "openai/gpt-5-mini (target)",
        "judge": JUDGE,
        "n_pairs": n_orig,
        "scoring": "LLM-as-judge",
        "orig_accuracy": n_orig_c / n_orig if n_orig else 0,
        "v3_accuracy": n_v3_c / n_v3 if n_v3 else 0,
        "delta": (n_v3_c / n_v3) - (n_orig_c / n_orig) if (n_orig and n_v3) else 0,
        "relative_preservation": n_v3_c / n_orig_c if n_orig_c else 0,
        "per_task": {t: {"orig_acc": (s["orig_c"]/s["orig_n"]) if s["orig_n"] else None,
                          "v3_acc": (s["v3_c"]/s["v3_n"]) if s["v3_n"] else None,
                          "orig_n": s["orig_n"], "v3_n": s["v3_n"]}
                      for t, s in by_task.items()},
    }
    (REPO / "results" / "task_metric_bbh_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {REPO / 'results' / 'task_metric_bbh_summary.json'}")


if __name__ == "__main__":
    main()
