"""Load (transcript, condition, sample_id) triples from existing parent-repo JSONs.

Conditions:
  orig_eval     — original E-cases (high_confidence_eval_cases ∩ transcripts)
  orig_deploy   — original D-cases (low_confidence_deploy_cases ∩ transcripts)
  v1_removed    — original aggressive sanitization
  v2_removed    — scaffolding-only filter
  v3_removed    — scaffolding-only + preservation gate
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

from activation_probing import config


def _load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Per-benchmark original transcripts
# ---------------------------------------------------------------------------

def _hf_tells_repo() -> Path:
    """Resolve the hf-tells-repo path. Prefers env var, falls back to known sibling location."""
    env = os.environ.get("HF_TELLS_REPO")
    if env:
        return Path(env)
    # Common layout: hf-tells-repo is a sibling of netzer-eval-awareness-tells
    sibling = config.REPO_ROOT.parent / "hf-tells-repo"
    if sibling.exists():
        return sibling
    return config.REPO_ROOT / "hf-tells-repo"


def _load_originals_for_benchmark(benchmark: str) -> dict[str, list[dict]]:
    """Return {sample_id: messages} for the benchmark's original transcripts."""
    repo = config.REPO_ROOT

    if benchmark == "needham":
        # Needham transcripts are gated; load via the parent repo's data_config
        os.environ.setdefault("EVAL_AWARENESS_DATA_FILE", "/tmp/stub_data.json")
        os.environ["HF_TELLS_REPO"] = str(_hf_tells_repo())
        sys.path.insert(0, str(repo))
        try:
            from tells.data_config import load_gated_transcripts
            ts = load_gated_transcripts()
            return {sid: t["input"] for sid, t in ts.items()}
        except Exception as exc:
            print(f"  [warn] Needham load failed: {exc}")
            return {}

    path = repo / "tells" / "data" / benchmark / "transcripts.json"
    items = _load_json(path)
    out = {}
    for t in items:
        out[t["id"]] = t["input"]
    return out


# ---------------------------------------------------------------------------
# E-case / D-case ID lookups
# ---------------------------------------------------------------------------

def _e_d_ids(benchmark: str) -> tuple[set[str], set[str]]:
    """Return (e_ids, d_ids) for the benchmark."""
    repo = config.REPO_ROOT

    if benchmark == "needham":
        hf = _hf_tells_repo()
        e = _load_json(hf / "metadata" / "source_cases" / "high_confidence_eval_cases.json")
        d = _load_json(hf / "metadata" / "source_cases" / "low_confidence_deploy_cases.json")
    else:
        # Prefer _full files (more cases), fall back to top-K
        e_full = repo / "tells" / "data" / benchmark / "high_confidence_eval_cases_full.json"
        e = _load_json(e_full) if e_full.exists() else _load_json(
            repo / "tells" / "data" / benchmark / "high_confidence_eval_cases.json"
        )
        d = _load_json(repo / "tells" / "data" / benchmark / "low_confidence_deploy_cases.json")

    e_ids = {c["sample_id"] for c in e}
    d_ids = {c["sample_id"] for c in d}
    return e_ids, d_ids


# ---------------------------------------------------------------------------
# Sanitized transcript paths
# ---------------------------------------------------------------------------

def _sanitized_paths(benchmark: str, version: str) -> Path | None:
    """Return path to a JSON file containing __removed-suffixed entries for this version."""
    repo = config.REPO_ROOT

    if version == "v1":
        # Original aggressive sanitization paths
        if benchmark == "needham":
            return repo / "tells" / "data" / "reverse" / "dataset.json"
        if benchmark in ("hawthorne", "mask"):
            return repo / "tells" / "data" / benchmark / "reverse" / "dataset.json"
        if benchmark == "agentharm":
            return repo / "tells" / "data" / benchmark / "reverse_native" / "removed.json"
        # bbh, malt
        return repo / "tells" / "data" / benchmark / "reverse" / "removed.json"

    if version == "v2":
        if benchmark == "needham":
            return repo / "tells" / "data" / "reverse_v2" / "removed.json"
        if benchmark == "sad":
            return repo / "tells" / "data" / "heldout_audit" / "v2" / "sad_removed.json"
        return repo / "tells" / "data" / benchmark / "reverse_v2" / "removed.json"

    if version == "v3":
        if benchmark == "needham":
            return repo / "tells" / "data" / "reverse_gated" / "removed.json"
        return repo / "tells" / "data" / benchmark / "reverse_gated" / "removed.json"

    raise ValueError(f"unknown version: {version}")


def _load_sanitized(benchmark: str, version: str) -> dict[str, list[dict]]:
    """Return {original_sample_id: sanitized messages} for the version."""
    path = _sanitized_paths(benchmark, version)
    if path is None or not path.exists():
        return {}
    items = _load_json(path)
    out = {}
    for s in items:
        sid = s["id"]
        # SAD heldout-audit uses __heldout_removed suffix
        if "__heldout_removed" in sid:
            base = sid.rsplit("__heldout_removed", 1)[0]
            out[base] = s["input"]
            continue
        if "__removed" not in sid:
            continue
        # Strip __removed (and __hyp__removed for hawthorne baseline naming)
        base = sid.rsplit("__removed", 1)[0]
        out[base] = s["input"]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_condition(
    benchmark: str,
    condition: str,
    n: int | None = None,
) -> list[tuple[str, list[dict]]]:
    """Return list of (sample_id, messages) for the given (benchmark, condition).

    Conditions:
      orig_eval     — original E-cases
      orig_deploy   — original D-cases
      v1_removed    — v1 aggressive sanitized E-cases
      v2_removed    — v2 scaffolding-only sanitized E-cases
      v3_removed    — v3 (with gate) sanitized E-cases
    """
    originals = _load_originals_for_benchmark(benchmark)
    e_ids, d_ids = _e_d_ids(benchmark)

    if condition == "orig_eval":
        ids = sorted(set(originals) & e_ids)
        out = [(sid, originals[sid]) for sid in ids]
    elif condition == "orig_deploy":
        ids = sorted(set(originals) & d_ids)
        out = [(sid, originals[sid]) for sid in ids]
    elif condition.endswith("_removed"):
        version = condition.split("_")[0]  # v1 / v2 / v3
        sanitized = _load_sanitized(benchmark, version)
        # Restrict to those whose original is an E-case
        ids = sorted(set(sanitized) & e_ids)
        out = [(sid, sanitized[sid]) for sid in ids]
    elif condition == "rev_control":
        # Matched-scope neutral rewrite controls (length-matched random-span deletions).
        # Same eval transcripts as v3, but with non-bank-targeted edits.
        repo = config.REPO_ROOT
        if benchmark == "needham":
            path = repo / "tells" / "data" / "reverse" / "control.json"
        elif benchmark == "sad":
            path = repo / "tells" / "data" / "heldout" / "reverse" / "control.json"
        else:
            path = repo / "tells" / "data" / benchmark / "reverse" / "control.json"
        items = _load_json(path) if path.exists() else []
        out = []
        for s in items:
            sid = s["id"]
            # Filter to SAD samples only when benchmark=sad (heldout has both SAD+AISI)
            if benchmark == "sad" and not sid.startswith("sad:"):
                continue
            base = sid.rsplit("__rev_ctrl", 1)[0]
            out.append((base, s["input"]))
    elif condition.startswith("fwd_"):
        # Forward injection conditions. Available subtypes:
        #   fwd_control          — control (no injection) from forward_paired/control_combined.json
        #   fwd_combined         — all tells injected from forward_paired/treatment_combined.json
        #   fwd_solo:<template>  — single template injected from forward_paired/treatment_<template>.json
        repo = config.REPO_ROOT
        ff_dir = repo / "tells" / "data" / benchmark / "forward_paired"
        if condition == "fwd_control":
            path = ff_dir / "control_combined.json"
        elif condition == "fwd_combined":
            path = ff_dir / "treatment_combined.json"
        elif condition.startswith("fwd_solo:"):
            tmpl = condition.split("fwd_solo:", 1)[1]
            path = ff_dir / f"treatment_{tmpl}.json"
        else:
            raise ValueError(f"unknown fwd condition: {condition}")
        items = _load_json(path)
        out = [(s["id"], s["input"]) for s in items]
    else:
        raise ValueError(f"unknown condition: {condition}")

    if n is not None:
        out = out[:n]
    return out


def coverage_summary() -> None:
    """Print sample counts per (benchmark, condition) — useful sanity check."""
    rows = []
    for b in config.BENCHMARKS:
        for c in config.CONDITIONS:
            try:
                items = load_condition(b, c)
                rows.append((b, c, len(items)))
            except Exception as exc:
                rows.append((b, c, f"ERR: {exc}"))
    print(f"{'benchmark':<12} {'condition':<14} {'n':>5}")
    print("-" * 40)
    for b, c, n in rows:
        print(f"{b:<12} {c:<14} {n!s:>5}")


if __name__ == "__main__":
    coverage_summary()
