"""Optional full-size Qwen3.5 Q4 logit check against a CPU Transformers model.

Needs about 20-25 GiB available host RAM. Model construction uses meta tensors;
weights are unpacked a few rows at a time. No original model files are modified.
"""
import argparse
import json
import subprocess
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, default=Path("vendor/models/MiMo-Q4"))
args = parser.parse_args()
torch.set_num_threads(8)
manifest = json.loads((args.model / "quantization.json").read_text())
raw_config = json.loads((args.model / "config.json").read_text())
config = Qwen3_5TextConfig(**raw_config["text_config"])
config._attn_implementation = "eager"
with torch.device("meta"):
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
weights = {}
for i, (name, spec) in enumerate(manifest["tensors"].items()):
    tensors = load_file(args.model / spec["file"])
    if spec["quantized"]:
        rows, cols = spec["shape"]
        output = torch.empty((rows, cols), dtype=torch.bfloat16)
        group = torch.arange(cols) // manifest["group_size"]
        for start in range(0, rows, 512):
            stop = min(start + 512, rows)
            codes = tensors[name][start:stop]
            codes = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(stop-start, -1)[:, :cols]
            scales = tensors[name + ".grout_scales"][start:stop]
            output[start:stop] = (codes.float()*scales[:, group, 0]+scales[:, group, 1]).to(torch.bfloat16)
    else:
        output = tensors[name].to(torch.bfloat16)
    weights[name.replace("model.language_model.", "model.")] = output
    if i % 50 == 0:
        print(f"Unpacked {i}/{len(manifest['tensors'])} tensors", flush=True)
model.load_state_dict(weights, strict=True, assign=True)
del weights, tensors, output
# RoPE buffers initialized on meta are non-persistent: instantiate the reference
# rotary module normally after loading parameters.
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
model.eval()
prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|><|im_start|>assistant\n<think></think>"
ids = Tokenizer.from_file(str(args.model / "tokenizer.json")).encode(prompt).ids
root = Path("target/quantized-real-oracle")
root.mkdir(parents=True, exist_ok=True)
native_output = root / "native.json"
subprocess.run(["target/release/grout_quant_probe.exe", "--model", str(args.model),
                "--tokens", ",".join(map(str, ids)), "--last-only", "--output", str(native_output)], check=True)
actual = torch.tensor(json.loads(native_output.read_text())["logits"][0])
print("Running full-size CPU reference forward", flush=True)
with torch.no_grad():
    expected = model(torch.tensor([ids]), use_cache=False).logits[0, -1].float()
delta = (actual-expected).abs()
# Deep BF16 implementations accumulate different rounding. Report the measured
# error and require the independently computed greedy token and strong agreement.
cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
assert actual.argmax().item() == expected.argmax().item(), "full-model greedy token differs"
assert cosine > 0.999, cosine
report = {"tokens": ids, "native_top1": actual.argmax().item(), "reference_top1": expected.argmax().item(),
          "max_logit_error": delta.max().item(), "mean_logit_error": delta.mean().item(), "cosine_similarity": cosine}
(root / "comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
