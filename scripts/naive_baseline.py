"""Naive baseline classifier comparison.

For each benchmark with both E and D cases:
  1. Train TF-IDF + logistic regression on (E, D) with 5-fold CV.
  2. Get out-of-fold P(eval) for original E-cases (=baseline classifier P).
  3. Train on full E+D, predict on sanitized E (=removed P) and control E (=ctrl P).
  4. Report delta vs the LLM judge delta from RESULTS.

Question: does a trivial bag-of-words classifier reproduce the LLM judge's
sanitization effect? If yes, our tells are picked up by even simple models —
useful signal but means LLM judge isn't doing anything sophisticated. If no,
the LLM judge is using semantic features the bag-of-words model misses, which
strengthens the "real audit tooling" framing.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('EVAL_AWARENESS_DATA_FILE', '/tmp/stub_data.json')
os.environ.setdefault('HF_TELLS_REPO', os.environ.get('HF_TELLS_REPO', 'hf-tells-repo'))
sys.path.insert(0, '.')

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def transcript_to_text(messages):
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content)
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def load_benchmark(name):
    """Returns dict: {id: {messages, eval_category}} for E+D, plus paths to reverse data."""
    if name == "hawthorne":
        e_cases = json.load(open("tells/data/hawthorne/high_confidence_eval_cases_full.json"))
        d_cases = json.load(open("tells/data/hawthorne/low_confidence_deploy_cases.json"))
        ts = {t["id"]: t for t in json.load(open("tells/data/hawthorne/transcripts.json"))}
        reverse_path = "tells/data/hawthorne/reverse/dataset.json"
    elif name == "mask":
        e_cases = json.load(open("tells/data/mask/high_confidence_eval_cases_full.json"))
        d_cases = json.load(open("tells/data/mask/low_confidence_deploy_cases.json"))
        ts = {t["id"]: t for t in json.load(open("tells/data/mask/transcripts.json"))}
        reverse_path = "tells/data/mask/reverse/dataset.json"
    elif name == "needham":
        from tells.data_config import load_gated_transcripts
        ts = load_gated_transcripts()
        e_cases = json.load(open("hf-tells-repo/metadata/source_cases/high_confidence_eval_cases.json"))
        d_cases = json.load(open("hf-tells-repo/metadata/source_cases/low_confidence_deploy_cases.json"))
        reverse_path = "tells/data/reverse/dataset.json"
    else:
        raise ValueError(name)

    e_ids = [c["sample_id"] for c in e_cases if c["sample_id"] in ts]
    d_ids = [c["sample_id"] for c in d_cases if c["sample_id"] in ts]
    return {
        "e_ids": e_ids,
        "d_ids": d_ids,
        "transcripts": ts,
        "reverse_path": reverse_path,
    }


def run_benchmark(name):
    print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
    bench = load_benchmark(name)
    ts = bench["transcripts"]
    e_ids, d_ids = bench["e_ids"], bench["d_ids"]
    print(f"  E={len(e_ids)} D={len(d_ids)}")

    # Build training corpus
    e_texts = [transcript_to_text(ts[i]["input"]) for i in e_ids]
    d_texts = [transcript_to_text(ts[i]["input"]) for i in d_ids]
    X_text = e_texts + d_texts
    y = np.array([1] * len(e_texts) + [0] * len(d_texts))

    # 5-fold CV out-of-fold predictions for E
    vec_proto = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p_eval_e = np.zeros(len(e_texts))
    oof_p_eval_d = np.zeros(len(d_texts))
    e_idx = np.arange(len(e_texts))
    d_idx = np.arange(len(e_texts), len(e_texts) + len(d_texts))
    all_idx = np.concatenate([e_idx, d_idx])
    for tr, te in skf.split(all_idx, y):
        tr_text = [X_text[i] for i in tr]
        te_text = [X_text[i] for i in te]
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(tr_text)
        Xte = vec.transform(te_text)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        for k, i in enumerate(te):
            if i < len(e_texts):
                oof_p_eval_e[i] = proba[k]
            else:
                oof_p_eval_d[i - len(e_texts)] = proba[k]

    base_e_p = float(np.mean(oof_p_eval_e))
    base_d_p = float(np.mean(oof_p_eval_d))
    auroc_proxy = float(np.mean([1 if oe > od else 0 for oe in oof_p_eval_e for od in oof_p_eval_d]))
    print(f"  Classifier baseline:  P(eval | E)={base_e_p:.3f}, P(eval | D)={base_d_p:.3f}")
    print(f"  Pairwise E>D rate (rough AUROC proxy): {auroc_proxy:.3f}")

    # Train on full data, predict on sanitized + control E
    vec_full = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True)
    X_full = vec_full.fit_transform(X_text)
    clf_full = LogisticRegression(max_iter=2000, C=1.0)
    clf_full.fit(X_full, y)

    # Load reverse dataset
    rev_ds = json.load(open(bench["reverse_path"]))
    by_suffix = {"baseline": [], "removed": [], "rev_ctrl": []}
    BASELINE_SUFFIXES = {"hyp", "baseline"}
    for s in rev_ds:
        sid = s["id"]
        suff = sid.split("__")[-1] if "__" in sid else "baseline"
        if suff in BASELINE_SUFFIXES:
            by_suffix["baseline"].append(s)
        elif suff == "removed":
            by_suffix["removed"].append(s)
        elif suff == "rev_ctrl":
            by_suffix["rev_ctrl"].append(s)

    print(f"  Reverse dataset: baseline={len(by_suffix['baseline'])}, removed={len(by_suffix['removed'])}, rev_ctrl={len(by_suffix['rev_ctrl'])}")

    if not by_suffix['removed']:
        print("  No 'removed' samples in reverse — skipping")
        return

    results = {}
    for suff in ["baseline", "removed", "rev_ctrl"]:
        if not by_suffix[suff]:
            continue
        texts = [transcript_to_text(s["input"]) for s in by_suffix[suff]]
        Xs = vec_full.transform(texts)
        ps = clf_full.predict_proba(Xs)[:, 1]
        results[suff] = (float(np.mean(ps)), float(np.std(ps)))

    print(f"  Sanitization probe (classifier P(eval)):")
    for suff, (mean, std) in results.items():
        print(f"    {suff:12s}: {mean:.3f} ± {std:.3f}")

    if "baseline" in results and "removed" in results:
        delta = results["removed"][0] - results["baseline"][0]
        print(f"  Classifier delta (removed - baseline): {delta:+.3f}")
    if "baseline" in results and "rev_ctrl" in results:
        ctrl_delta = results["rev_ctrl"][0] - results["baseline"][0]
        print(f"  Classifier delta (control - baseline):  {ctrl_delta:+.3f}")
    if "removed" in results and "rev_ctrl" in results:
        rc_delta = results["removed"][0] - results["rev_ctrl"][0]
        print(f"  Classifier causal (removed - control):  {rc_delta:+.3f}  <-- the controlled effect")

    return {
        "benchmark": name,
        "n_e": len(e_ids),
        "n_d": len(d_ids),
        "classifier_oof_p_e": base_e_p,
        "classifier_oof_p_d": base_d_p,
        "auroc_proxy": auroc_proxy,
        "reverse_p_eval": results,
    }


if __name__ == "__main__":
    out = []
    for b in ["hawthorne", "mask", "needham"]:
        try:
            r = run_benchmark(b)
            if r:
                out.append(r)
        except Exception as e:
            import traceback
            print(f"\nERROR on {b}:")
            traceback.print_exc()
    Path("tells/data/naive_baseline_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n\nWrote results to tells/data/naive_baseline_results.json")
