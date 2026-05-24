"""Forward-injection cell #11: bbh→haw causal Δ.

Adds the missing cell to complete the 11/11 cross-injection table.
Pattern: take BBH templates, inject into Hawthorne D-cases, score baseline / treatment / control.

Reuses the cross_inject_expansion.py runner pattern but for the single bbh→haw cell.
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

from openai import AsyncOpenAI

# Reuse logic from cross_inject_expansion.py
from cross_inject_expansion import (
    INJECT_PROMPT, CONTROL_PROMPT, fmt, apply_diff,
    call_writer, score, load_source_templates, load_target_d_cases,
    format_templates_for_prompt,
)


async def main():
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    write_sem = asyncio.Semaphore(8)
    score_sem = asyncio.Semaphore(16)

    # Add bbh source loader since cross_inject_expansion.py doesn't have it
    bbh_templates = json.load(open("tells/data/bbh/tell_templates.json"))
    bbh_templates.sort(key=lambda t: -(t.get("frequency") or 0))
    templates = bbh_templates[:3]
    print(f"BBH templates: {[t.get('template_id', t.get('id', '?')) for t in templates]}")

    # Hawthorne D-cases
    haw_transcripts = {t["id"]: t for t in json.load(open("tells/data/hawthorne/transcripts.json"))}
    haw_d = json.load(open("tells/data/hawthorne/low_confidence_deploy_cases.json"))
    d_cases = [haw_transcripts[c["sample_id"]] for c in haw_d if c["sample_id"] in haw_transcripts][:20]
    print(f"Hawthorne D-cases: {len(d_cases)}")

    templates_text = format_templates_for_prompt(templates)

    async def gen_treatment(d):
        prompt = INJECT_PROMPT.format(
            templates_text=templates_text,
            transcript_json=json.dumps(fmt(d["input"]), indent=2)[:10000],
        )
        result = await call_writer(client, write_sem, prompt, d["id"])
        if result is None or result.get("incompatible"):
            return None
        try:
            return apply_diff(d["input"], result)
        except Exception:
            return None

    async def gen_control(d):
        prompt = CONTROL_PROMPT.format(transcript_json=json.dumps(fmt(d["input"]), indent=2)[:10000])
        result = await call_writer(client, write_sem, prompt, d["id"])
        if result and result.get("messages"):
            return result["messages"]
        return None

    print("Generating treatments + controls...")
    treatments = await asyncio.gather(*[gen_treatment(d) for d in d_cases])
    controls = await asyncio.gather(*[gen_control(d) for d in d_cases])

    print(f"  treatments OK: {sum(1 for t in treatments if t)}/{len(d_cases)}")
    print(f"  controls OK:   {sum(1 for c in controls if c)}/{len(d_cases)}")

    print("Scoring baseline + treatment + control...")
    score_tasks = []
    for i, d in enumerate(d_cases):
        score_tasks.append(("baseline", i, score(client, score_sem, d["input"])))
        if treatments[i]:
            score_tasks.append(("treatment", i, score(client, score_sem, treatments[i])))
        if controls[i]:
            score_tasks.append(("control", i, score(client, score_sem, controls[i])))
    raw = await asyncio.gather(*[t[2] for t in score_tasks])
    scores = {}
    for (kind, i, _), s in zip(score_tasks, raw):
        scores.setdefault(i, {})[kind] = s

    causals = []
    treats = []; ctrls = []
    for i in range(len(d_cases)):
        s = scores.get(i, {})
        t, c, b = s.get("treatment"), s.get("control"), s.get("baseline")
        if t is None or c is None:
            continue
        causals.append(t - c)
        treats.append(t); ctrls.append(c)

    if not causals:
        print("ZERO valid paired (treatment, control) — abort")
        return

    n = len(causals)
    from statistics import mean, stdev
    mean_d = mean(causals)
    sd_d = stdev(causals) if n > 1 else 0
    t_stat = mean_d / (sd_d / (n ** 0.5)) if sd_d > 0 else 0
    treat_mean = mean(treats)
    ctrl_mean = mean(ctrls)

    print()
    print("=" * 60)
    print(f"BBH → HAW (n={n})")
    print(f"  treat mean: {treat_mean:+.3f}")
    print(f"  ctrl mean:  {ctrl_mean:+.3f}")
    print(f"  causal Δ:   {mean_d:+.3f}")
    print(f"  t-stat:     {t_stat:+.2f}")

    out = {
        "source": "bbh", "target": "hawthorne", "n": n,
        "treat_mean": treat_mean, "ctrl_mean": ctrl_mean,
        "causal_mean": mean_d, "t_stat": t_stat, "sd_d": sd_d,
    }
    Path("/tmp/bbh_to_haw_cell.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] /tmp/bbh_to_haw_cell.json")


if __name__ == "__main__":
    asyncio.run(main())
