"""Compare native packed Qwen3 forward/loss/every gradient with HF/PyTorch.

First run the ignored Rust export_training_oracle test. This validates actual
tensor operations against an independent implementation, including RoPE and GQA.
"""
import json
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

root = Path("target/mtp-training-oracle")
fixture = json.loads((root / "oracle.json").read_text())
config = Qwen3Config.from_pretrained(root)
model = Qwen3ForCausalLM.from_pretrained(root, config=config, dtype=torch.float32, attn_implementation="eager")
teacher = Qwen3ForCausalLM.from_pretrained(root, config=config, dtype=torch.float32, attn_implementation="eager")
teacher.requires_grad_(False)
model.eval()
teacher.eval()
tokens = torch.tensor([fixture["tokens"]])
positions = torch.tensor([fixture["positions"]])
visible = torch.tensor(fixture["visible"])
mask = torch.where(visible, 0.0, float("-inf"))[None, None]
rows = fixture["rows"]
logits = model(input_ids=tokens, position_ids=positions, attention_mask=mask, use_cache=False).logits[0, rows]
torch.testing.assert_close(logits, torch.tensor(fixture["logits"]), atol=2e-5, rtol=2e-4)
with torch.no_grad():
    proposals = logits.argmax(-1).reshape(-1, fixture["k"])
    forcing = tokens.clone()
    forcing[0, fixture["mask_rows"]] = proposals[:, :-1].flatten()
    targets = teacher(input_ids=forcing, position_ids=positions, attention_mask=mask, use_cache=False).logits[0, rows].argmax(-1)
loss = torch.nn.functional.cross_entropy(logits, targets)
torch.testing.assert_close(loss.detach(), torch.tensor(fixture["hard_loss"]), atol=2e-5, rtol=2e-4)
loss.backward()
max_error = 0.0
for name, parameter in model.named_parameters():
    expected = torch.tensor(fixture["gradients"][name]).reshape(parameter.shape)
    torch.testing.assert_close(parameter.grad, expected, atol=3e-5, rtol=3e-3)
    max_error = max(max_error, (parameter.grad - expected).abs().max().item())
report = {"logits_match": True, "hard_loss_match": True, "all_parameter_gradients_match": True,
          "max_gradient_absolute_error": max_error, "loss": loss.item()}
(root / "comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
