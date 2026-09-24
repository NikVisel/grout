"""Optional oracle check. Python is used only for validation, never by Grout.

Install transformers==4.57.3 and safetensors in a test environment containing
PyTorch. --reference-code points at the model repository's pinned Python files.
"""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference-code", type=Path, required=True)
    p.add_argument("--native-json", type=Path, required=True)
    p.add_argument("--prompt", default="What is 2 + 2?")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--threshold", type=float)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    # Import the authors' reviewed, pinned files without changing the model dir.
    package = types.ModuleType("mtp_reference")
    package.__path__ = [str(args.reference_code.resolve())]
    sys.modules[package.__name__] = package
    module = importlib.import_module("mtp_reference.modeling_qwen3")
    model = module.Qwen3ForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    result = model.generate(
        input_ids=ids, do_mtp=True, k_toks=args.k, mask_id=151669,
        eos_id=[151645, 151643], max_returned_tokens=ids.shape[1] + args.max_new_tokens,
        strategy=None if args.threshold is None else ["conf_adapt", args.threshold],
        return_mtp_result_dict=True, include_prompt=False,
    )
    reference = result["token_ids"][0].tolist()
    # Authors stop after the whole block; Grout deliberately stops at the first
    # EOS within it. Compare the same meaningful prefix including that EOS.
    for i, token in enumerate(reference):
        if token in [151645, 151643]:
            reference = reference[:i + 1]
            break
    native = json.loads(args.native_json.read_text(encoding="utf-8"))
    report = {"reference_tokens": reference, "native_tokens": native["token_ids"],
              "reference_chunks": result["effective_k_values"], "native_chunks": native["mtp_chunks"],
              "tokens_match": reference == native["token_ids"], "reference_dtype": "float32",
              "text": tokenizer.decode(reference, skip_special_tokens=True)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert report["tokens_match"], "Native output differs; inspect the saved report."


if __name__ == "__main__":
    main()
