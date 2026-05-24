# Needham benchmark data

This benchmark is **gated**. To reproduce experiments that depend on
Needham E/D-cases, request access to `jjpn2/eval_awareness` on
HuggingFace and set the following in `.env`:

    HF_TOKEN=hf_<your-token>
    HF_TELLS_REPO=/path/to/cloned/hf-tells-repo

`bank_pipeline/data_config.py` will then resolve Needham transcripts via
the HuggingFace API at runtime.
