"""(1) Training-data audit.

Characterize the safety-vs-capability composition of the probe training set.
For each benchmark in the training set (needham, hawthorne, mask), tag as
capability or safety based on benchmark provenance + content keyword density.

If safety dominates, that's the smoking gun for the 'probe is a safety detector'
critique.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.data_loader import load_condition


# Benchmark provenance: hand-coded based on benchmark design intent
BENCHMARK_TYPE = {
    "needham": "capability",      # n-back, AppWorld — agentic reasoning
    "bbh": "capability",          # Big-Bench Hard — multi-task reasoning
    "malt": "capability",         # MALT — multi-agent code
    "hawthorne": "safety",        # harmful-request refusal
    "mask": "safety",             # honesty / deception
    "agentharm": "safety",        # harmful agent tools
    "sad": "situational-awareness",  # SAD
    "aisi": "safety",
}

# Keywords indicating safety/harm content
SAFETY_KEYWORDS = [
    "refus", "harm", "danger", "unsafe", "policy", "guideline", "violat",
    "ethic", "illegal", "weapon", "explos", "drug", "abuse", "exploit",
    "malicious", "fraud", "scam", "manipulat", "deceiv", "decept", "lie",
    "phish", "hack", "attack", "vulnerab", "credential", "password",
    "kill", "murder", "suicid", "self-harm", "child", "minor", "porn",
    "racist", "discriminat", "hate", "bias", "stereo", "propagand",
    "disinform", "misinform", "fake news",
]

# Keywords indicating capability/reasoning content
CAPABILITY_KEYWORDS = [
    "calculate", "compute", "solve", "logic", "deduc", "infer", "step",
    "math", "algebra", "equation", "function", "algorithm", "code",
    "program", "debug", "implement", "answer", "question", "puzzle",
    "reason", "prove", "theorem", "result", "output", "expected",
]


def keyword_density(text: str, keywords: list[str]) -> float:
    """Return fraction of words matching any keyword pattern."""
    text_l = text.lower()
    words = text_l.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if any(kw in w for kw in keywords))
    return hits / len(words)


def transcript_text(msgs) -> str:
    parts = []
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
        parts.append(str(c))
    return " ".join(parts)


def audit_benchmark(benchmark: str) -> dict:
    """For each (benchmark, condition), compute n + safety/capability density."""
    out = {"benchmark": benchmark, "type": BENCHMARK_TYPE.get(benchmark, "unknown")}
    for cond in ["orig_eval", "orig_deploy"]:
        try:
            items = load_condition(benchmark, cond)
        except Exception:
            continue
        if not items:
            continue
        safety_dens = []
        cap_dens = []
        lengths = []
        for sid, msgs in items[:300]:  # cap at 300 for speed
            text = transcript_text(msgs)
            safety_dens.append(keyword_density(text, SAFETY_KEYWORDS))
            cap_dens.append(keyword_density(text, CAPABILITY_KEYWORDS))
            lengths.append(len(text.split()))
        if not safety_dens:
            continue
        out[cond] = {
            "n": len(safety_dens),
            "mean_word_count": round(sum(lengths) / len(lengths)),
            "safety_density_mean_pct": 100 * sum(safety_dens) / len(safety_dens),
            "capability_density_mean_pct": 100 * sum(cap_dens) / len(cap_dens),
        }
    return out


def main():
    benchmarks = ["needham", "hawthorne", "mask", "agentharm", "bbh", "malt", "sad"]
    rows = [audit_benchmark(b) for b in benchmarks]

    print("=" * 100)
    print("(1) Training-data audit — benchmark composition")
    print("-" * 100)
    print(f"{'benchmark':<12} {'type':<22} {'cond':<15} {'n':>5} {'words':>8} {'safety%':>10} {'capability%':>13}")
    print("-" * 100)
    for r in rows:
        for cond in ["orig_eval", "orig_deploy"]:
            d = r.get(cond)
            if not d:
                continue
            print(f"{r['benchmark']:<12} {r['type']:<22} {cond:<15} {d['n']:>5} {d['mean_word_count']:>8} "
                  f"{d['safety_density_mean_pct']:>10.2f} {d['capability_density_mean_pct']:>13.2f}")

    print()
    print("=" * 100)
    print("Training set summary (probes trained on needham + hawthorne + mask only)")
    print("-" * 100)
    train_bms = [r for r in rows if r["benchmark"] in ["needham", "hawthorne", "mask"]]
    train_types = Counter(r["type"] for r in train_bms)
    print("Benchmarks in training set by type:")
    for t, c in train_types.items():
        print(f"  {t}: {c}")
    train_total_n = 0
    train_safety_n = 0
    train_capability_n = 0
    for r in train_bms:
        for cond in ["orig_eval", "orig_deploy"]:
            d = r.get(cond)
            if not d:
                continue
            train_total_n += d["n"]
            if r["type"] == "safety":
                train_safety_n += d["n"]
            elif r["type"] == "capability":
                train_capability_n += d["n"]
    print(f"\nTotal samples in pooled training set: {train_total_n}")
    print(f"  Safety: {train_safety_n} ({100*train_safety_n/train_total_n:.0f}%)")
    print(f"  Capability: {train_capability_n} ({100*train_capability_n/train_total_n:.0f}%)")

    # Save
    out = config.RESULTS_DIR / "audit_training_data.json"
    out.write_text(json.dumps({
        "per_benchmark": rows,
        "training_set_summary": {
            "n_safety": train_safety_n,
            "n_capability": train_capability_n,
            "total": train_total_n,
        }
    }, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
