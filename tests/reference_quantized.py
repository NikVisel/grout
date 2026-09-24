"""Independent HF/PyTorch oracle for Q4 GPU logits, chunking and state rollback.

Run with target/qwen35-reference-venv/Scripts/python (transformers==5.12.1).
Only small deterministic fixtures are created under target. Production inference
does not depend on this script or Transformers.
"""
import json
import subprocess
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer, models
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3_5TextConfig, Qwen3_5ForCausalLM

ROOT = Path("target/quantized-oracle")
BIN = Path("target/release")
TOKENS = [1, 9, 3, 21, 7, 12, 2]


def run(*args):
    subprocess.run([str(x) for x in args], check=True)


def unpack(directory):
    manifest = json.loads((directory / "quantization.json").read_text())
    result = {}
    for name, spec in manifest["tensors"].items():
        tensors = load_file(directory / spec["file"])
        if spec["quantized"]:
            rows, cols = spec["shape"]
            codes = tensors[name]
            codes = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(rows, -1)[:, :cols]
            scales = tensors[name + ".grout_scales"]
            group = torch.arange(cols) // manifest["group_size"]
            result[name] = (codes.float() * scales[:, group, 0] + scales[:, group, 1]).to(torch.bfloat16)
        else:
            result[name] = tensors[name].to(torch.bfloat16)
    return result


def prepare(kind):
    source = ROOT / kind
    packed = ROOT / (kind + "-q4")
    if not source.exists():
        source.mkdir(parents=True)
        torch.manual_seed(713)
        common = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=64, eos_token_id=127,
                      tie_word_embeddings=False, attention_dropout=0.0)
        if kind == "qwen35":
            config = Qwen3_5TextConfig(**common, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                rope_parameters=dict(rope_type="default", rope_theta=10000000.0,
                                     partial_rotary_factor=0.5, mrope_section=[2, 1, 1], mrope_interleaved=True))
            model = Qwen3_5ForCausalLM(config)
        else:
            config = Qwen3Config(**common, use_sliding_window=False)
            model = Qwen3ForCausalLM(config)
        model.to(torch.bfloat16).save_pretrained(source)
        with safe_open(source / "model.safetensors", framework="pt") as f:
            index = {name: "model.safetensors" for name in f.keys()}
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
        vocab = {str(i): i for i in range(127)}
        vocab["<|mtp_special_token_0|>"] = 127
        Tokenizer(models.WordLevel(vocab, unk_token="0")).save(str(source / "tokenizer.json"))
    if not packed.exists():
        run(BIN / "grout_quantize.exe", "--model", source, "--output", packed, "--group-size", 16)
    return source, packed


def compare(kind):
    source, packed = prepare(kind)
    config = (Qwen3_5TextConfig if kind == "qwen35" else Qwen3Config).from_pretrained(source)
    config._attn_implementation = "eager"
    model = (Qwen3_5ForCausalLM if kind == "qwen35" else Qwen3ForCausalLM)(config).to(torch.bfloat16).eval()
    model.load_state_dict(unpack(packed), strict=True)
    with torch.no_grad():
        expected = model(torch.tensor([TOKENS]), use_cache=False).logits[0].float()
    results = []
    for label, chunks, temporary in [("full", [7], []), ("chunked", [2, 1, 4], []),
                                      ("rollback", [2, 1, 4], [31, 17])]:
        output = ROOT / f"{kind}-{label}.json"
        args = [BIN / "grout_quant_probe.exe", "--model", packed, "--tokens", ",".join(map(str, TOKENS)),
                "--chunks", ",".join(map(str, chunks)), "--output", output]
        if temporary:
            args += ["--temporary", ",".join(map(str, temporary))]
        run(*args)
        results.append(torch.tensor(json.loads(output.read_text())["logits"]))
    # BF16 rounding and reduction order differ across kernels; this bound is
    # substantially below a typical logit gap and checked together with argmax.
    torch.testing.assert_close(results[0], expected, atol=0.012, rtol=0.035)
    assert torch.equal(results[0].argmax(-1), expected.argmax(-1)), kind
    assert torch.equal(results[0], results[1]), "chunked cache changed logits"
    assert torch.equal(results[0], results[2]), "temporary states contaminated real tokens"
    return {"architecture": kind, "max_logit_error": (results[0]-expected).abs().max().item(),
            "argmax_match": True, "chunked_exact": True, "rollback_exact": True}


if __name__ == "__main__":
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    report = [compare(kind) for kind in ["qwen3", "qwen35"]]
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
