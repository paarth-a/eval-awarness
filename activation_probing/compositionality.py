"""Compositionality experiment runner.

For one benchmark, loads model + trained probe once, then for each forward-injection
condition (control, combined, per-template solo) extracts activations at the probe's
best layer and computes mean P(eval). Outputs:
  - results/compositionality_<benchmark>.json: per-condition mean P(eval) + N
  - results/compositionality_<benchmark>.md: human-readable table + linear-sum
    prediction vs actual combined Δ

Usage:
    ACTPROBE_MODEL=Qwen/Qwen3.5-27B python -m activation_probing.compositionality --benchmark hawthorne
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config, data_loader


def list_solo_templates(benchmark: str) -> list[str]:
    ff = config.REPO_ROOT / "tells" / "data" / benchmark / "forward_full"
    out = []
    for p in sorted(ff.glob("treatment_*.json")):
        name = p.stem.removeprefix("treatment_")
        if name == "combined":
            continue
        out.append(name)
    return out


def encode_one(messages, tokenizer, max_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return enc.input_ids.to("cuda")


def mean_p_eval(
    items: list,
    model,
    tokenizer,
    layer_idx: int,
    scaler,
    clf,
    max_tokens: int,
    batch_log_every: int = 25,
) -> tuple[float, int]:
    if not items:
        return float("nan"), 0
    captured = {}
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach()
    h_handle = model.model.layers[layer_idx].register_forward_hook(hook)
    p_evals = []
    try:
        with torch.no_grad():
            for i, (sid, msgs) in enumerate(items):
                input_ids = encode_one(msgs, tokenizer, max_tokens)
                model(input_ids=input_ids, use_cache=False)
                last = captured["h"][0, -1, :].to(torch.float32).cpu().numpy()
                xs = scaler.transform(last.reshape(1, -1))
                p = float(clf.predict_proba(xs)[0, 1])
                p_evals.append(p)
                if (i + 1) % batch_log_every == 0:
                    print(f"  ... {i+1}/{len(items)} done", flush=True)
    finally:
        h_handle.remove()
    return float(np.mean(p_evals)), len(p_evals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--n-per", type=int, default=int(config.N_PER_CONDITION or 150))
    args = ap.parse_args()

    # Load existing probe
    probe_path = config.PROBES_DIR / f"{args.benchmark}__probes.pkl"
    if not probe_path.exists():
        print(f"ERROR: probe not found at {probe_path}. Restore it from a backup first.")
        sys.exit(1)
    with open(probe_path, "rb") as f:
        probe = pickle.load(f)
    best_layer = int(probe["best_layer"])
    cv_acc = float(probe["best_layer_cv_acc"])
    print(f"Using probe trained at layer {best_layer}, CV acc={cv_acc:.3f}")

    # Discover conditions
    solos = list_solo_templates(args.benchmark)
    conditions = ["fwd_control", "fwd_combined"] + [f"fwd_solo:{t}" for t in solos]
    print(f"Conditions to run: {len(conditions)} ({len(solos)} solos + control + combined)")

    # Load model
    print(f"Loading {config.MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Run
    results = {}
    for cond in conditions:
        items = data_loader.load_condition(args.benchmark, cond, n=args.n_per)
        print(f"[{cond}] n={len(items)}", flush=True)
        mean, n = mean_p_eval(
            items, model, tokenizer, best_layer, probe["scaler"], probe["classifier"], config.MAX_TOKENS
        )
        results[cond] = {"mean_p_eval": mean, "n": n}
        print(f"  → mean P(eval) = {mean:.3f} (n={n})", flush=True)

    # Compositionality stats
    p_ctrl = results["fwd_control"]["mean_p_eval"]
    p_comb = results["fwd_combined"]["mean_p_eval"]
    solo_deltas = {}
    for t in solos:
        key = f"fwd_solo:{t}"
        if key in results:
            solo_deltas[t] = results[key]["mean_p_eval"] - p_ctrl
    combined_delta = p_comb - p_ctrl
    sum_solo = sum(solo_deltas.values())
    additivity_ratio = (combined_delta / sum_solo) if sum_solo != 0 else float("nan")

    out = {
        "benchmark": args.benchmark,
        "model": config.MODEL_NAME,
        "probe_layer": best_layer,
        "probe_cv_acc": cv_acc,
        "control_p_eval": p_ctrl,
        "combined_p_eval": p_comb,
        "combined_delta": combined_delta,
        "sum_solo_deltas": sum_solo,
        "additivity_ratio": additivity_ratio,
        "solo_deltas_sorted": sorted(solo_deltas.items(), key=lambda x: -x[1]),
        "per_condition": results,
    }
    out_json = config.RESULTS_DIR / f"compositionality_{args.benchmark}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_json}")

    # Markdown
    md_lines = [f"# Compositionality — {args.benchmark} on {config.MODEL_NAME}\n"]
    md_lines.append(f"Probe layer {best_layer}, CV acc {cv_acc:.3f}\n")
    md_lines.append(f"Control P(eval)  = {p_ctrl:.3f}")
    md_lines.append(f"Combined P(eval) = {p_comb:.3f}")
    md_lines.append(f"Combined Δ       = {combined_delta:+.3f}")
    md_lines.append(f"Sum of solo Δs   = {sum_solo:+.3f}")
    md_lines.append(f"**Additivity ratio (combined / sum) = {additivity_ratio:.2f}**")
    md_lines.append(
        "  - 1.0 = perfect linearity, <1 = saturation, >1 = superadditive\n"
    )
    md_lines.append("\n## Per-template solo Δs (sorted)\n")
    md_lines.append("| Template | Solo Δ |")
    md_lines.append("|---|---|")
    for tmpl, d in sorted(solo_deltas.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {tmpl} | {d:+.3f} |")
    out_md = config.RESULTS_DIR / f"compositionality_{args.benchmark}.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
