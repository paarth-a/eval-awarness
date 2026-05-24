"""Extract per-layer last-token activations from an open-weight model.

Usage:
    python -m activation_probing.extract_activations --benchmark hawthorne --condition orig_eval
    python -m activation_probing.extract_activations --all     # all benchmarks × all conditions

Output: cache/{benchmark}_{condition}.npz with arrays per layer + sample_ids.

Implementation notes:
- Uses register_forward_hook on transformer layer outputs (model.model.layers[i] for Llama/Qwen-style HF models)
- Captures activation at the LAST input token position (most informative for transcript classification)
- Uses fp16 for inference; saves activations as fp32 (probe training is sensitive)

Goodfire-API alternative path:
- If the user prefers Goodfire's hosted activation API instead of self-host, swap the `extract_one`
  function to call Goodfire's endpoint. Schema: send {model_id, prompt, layer_indices}, receive
  per-layer activation vectors. Not implemented here (TODO if user opts in).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as `python -m activation_probing.extract_activations`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activation_probing import config
from activation_probing.data_loader import load_condition


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_HOOKS = []
_BUFFER: dict[int, torch.Tensor] = {}


def load_model_and_tokenizer():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    print(f"Loading {config.MODEL_NAME} ...")
    _TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
    dtype = torch.float16 if config.DTYPE == "float16" else torch.bfloat16
    _MODEL = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # safer with hooks; can switch to "sdpa" if stable
    )
    _MODEL.eval()
    return _MODEL, _TOKENIZER


def get_layer_modules(model) -> list:
    """Return the list of decoder-layer modules. Works for Llama / Qwen / Mistral families."""
    # HF naming convention: model.model.layers (Llama, Qwen, Mistral, OLMo, ...)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # Some models use different attribute names; extend here if needed.
    raise RuntimeError(
        "Could not locate decoder layers. Expected model.model.layers; got "
        f"{type(model).__name__}. Patch get_layer_modules for this architecture."
    )


def select_layer_indices(n_layers: int) -> list[int]:
    if config.LAYER_INDICES is not None:
        return [i for i in config.LAYER_INDICES if 0 <= i < n_layers]
    # Auto-pick: every 4th layer + last 3
    auto = list(range(4, n_layers, 4)) + [n_layers - 3, n_layers - 2, n_layers - 1]
    return sorted(set(auto))


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------

def _make_hook(layer_idx: int):
    def hook(module, inputs, output):
        # output is a tuple (hidden_state, ...) for decoder layers
        h = output[0] if isinstance(output, tuple) else output
        # h shape: [batch, seq, hidden]; take last position
        _BUFFER[layer_idx] = h[:, -1, :].detach().to(torch.float32).cpu()
    return hook


def install_hooks(model, layer_indices: list[int]):
    global _HOOKS
    layers = get_layer_modules(model)
    for li in layer_indices:
        h = layers[li].register_forward_hook(_make_hook(li))
        _HOOKS.append(h)


def remove_hooks():
    global _HOOKS
    for h in _HOOKS:
        h.remove()
    _HOOKS = []


# ---------------------------------------------------------------------------
# Per-transcript activation extraction
# ---------------------------------------------------------------------------

def messages_to_input_ids(messages: list[dict], tokenizer, max_tokens: int):
    """Apply chat template, tokenize, truncate from the LEFT (keep last max_tokens).

    Truncating from the left preserves the most recent context, which is more relevant
    for "is this a continuing eval?" classification.
    """
    # Some models choke on tool messages / list contents — normalize to text.
    norm = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
            )
        if role == "tool":
            # Many tokenizers reject "tool" role; convert to a marker.
            role = "user"
            content = f"[tool result]\n{content}"
        norm.append({"role": role, "content": content if isinstance(content, str) else str(content)})

    try:
        text = tokenizer.apply_chat_template(
            norm, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: simple concatenation.
        text = "\n".join(f"[{m['role']}] {m['content']}" for m in norm)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        # Some tokenizers default to right-truncation; keep last max_tokens
        # by manually truncating before the encoder if needed.
    )
    if enc.input_ids.shape[-1] >= max_tokens:
        # Re-tokenize and truncate from the left manually
        full = tokenizer(text, return_tensors="pt")
        ids = full.input_ids[0, -max_tokens:]
        attn = full.attention_mask[0, -max_tokens:]
        enc = {"input_ids": ids.unsqueeze(0), "attention_mask": attn.unsqueeze(0)}
    return enc


@torch.no_grad()
def extract_one(model, tokenizer, messages: list[dict], layer_indices: list[int]) -> dict[int, np.ndarray]:
    enc = messages_to_input_ids(messages, tokenizer, config.MAX_TOKENS)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    _BUFFER.clear()
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return {li: _BUFFER[li][0].numpy() for li in layer_indices if li in _BUFFER}


# ---------------------------------------------------------------------------
# Per-(benchmark, condition) extraction
# ---------------------------------------------------------------------------

def cache_path(benchmark: str, condition: str) -> Path:
    safe_model = config.MODEL_NAME.replace("/", "__")
    return config.CACHE_DIR / f"{safe_model}__{benchmark}__{condition}.npz"


def run_extraction(benchmark: str, condition: str, n: int | None = None, overwrite: bool = False) -> Path:
    out = cache_path(benchmark, condition)
    if out.exists() and not overwrite:
        print(f"  [skip] {out.name} exists")
        return out

    items = load_condition(benchmark, condition, n=n or config.N_PER_CONDITION)
    if not items:
        print(f"  [empty] {benchmark}/{condition} — no data")
        return out
    print(f"  [{benchmark}/{condition}] n={len(items)}")

    model, tokenizer = load_model_and_tokenizer()
    layers = get_layer_modules(model)
    layer_indices = select_layer_indices(len(layers))
    if not _HOOKS:
        install_hooks(model, layer_indices)

    sample_ids = []
    per_layer = {li: [] for li in layer_indices}

    for sid, messages in tqdm(items, desc=f"{benchmark}/{condition}"):
        try:
            acts = extract_one(model, tokenizer, messages, layer_indices)
        except Exception as exc:
            print(f"    ERR [{sid}]: {exc}")
            continue
        sample_ids.append(sid)
        for li in layer_indices:
            if li in acts:
                per_layer[li].append(acts[li])

    if not sample_ids:
        print(f"  [empty] no successful extractions for {benchmark}/{condition}")
        return out

    save_dict = {f"layer_{li}": np.stack(per_layer[li], axis=0) for li in layer_indices if per_layer[li]}
    save_dict["sample_ids"] = np.array(sample_ids)
    save_dict["layer_indices"] = np.array(layer_indices)
    np.savez_compressed(out, **save_dict)
    print(f"  [saved] {out} (n={len(sample_ids)}, layers={len(layer_indices)})")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, default=None,
                    help="Single benchmark name (default: from config.BENCHMARKS via --all)")
    ap.add_argument("--condition", type=str, default=None,
                    help="Single condition name")
    ap.add_argument("--all", action="store_true",
                    help="Run all benchmarks × all conditions sequentially")
    ap.add_argument("--n", type=int, default=None,
                    help="Override config.N_PER_CONDITION (limit per cell)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if cache exists")
    args = ap.parse_args()

    if args.all:
        for b in config.BENCHMARKS:
            for c in config.CONDITIONS:
                try:
                    run_extraction(b, c, n=args.n, overwrite=args.overwrite)
                except Exception as exc:
                    print(f"  ERR running {b}/{c}: {exc}")
        remove_hooks()
        return

    if not args.benchmark or not args.condition:
        ap.error("Provide --benchmark and --condition (or use --all)")

    run_extraction(args.benchmark, args.condition, n=args.n, overwrite=args.overwrite)
    remove_hooks()


if __name__ == "__main__":
    main()
