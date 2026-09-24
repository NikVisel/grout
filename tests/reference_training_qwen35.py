"""Check Rust Qwen3.5 forward, packed MTP loss, and every gradient against HF.

Run the ignored export_qwen35_training_oracle test first. The independent model
replays each visible causal branch as its own sequence; it does not reuse the
Rust branch scheduler or recurrent implementation.
"""
import json
from pathlib import Path
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

torch.set_num_threads(4)
root = Path("target/qwen35-training-oracle")
fixture = json.loads((root / "oracle.json").read_text())
config = Qwen3_5TextConfig.from_pretrained(root)
config._attn_implementation = "eager"
student = Qwen3_5ForCausalLM.from_pretrained(root, config=config, dtype=torch.float32).eval()
teacher = Qwen3_5ForCausalLM.from_pretrained(root, config=config, dtype=torch.float32).eval()
teacher.requires_grad_(False)


def branches(model, tokens):
    result = []
    for row in fixture["rows"]:
        chain = [i for i, visible in enumerate(fixture["visible"][row]) if visible]
        ids = torch.tensor([[tokens[i] for i in chain]])
        positions = torch.tensor([[fixture["positions"][i] for i in chain]])
        result.append(model(ids, position_ids=positions, use_cache=False).logits[0, -1])
    return torch.stack(result)


with torch.no_grad():
    ntp = student(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False).logits[0]
    torch.testing.assert_close(ntp, torch.tensor(fixture["ntp_logits"]), atol=2e-5, rtol=3e-4)
    target = branches(teacher, fixture["teacher_tokens"]).argmax(-1)
    assert target.tolist() == fixture["targets"]
logits = branches(student, fixture["tokens"])
torch.testing.assert_close(logits, torch.tensor(fixture["logits"]), atol=2e-5, rtol=3e-4)
loss = torch.nn.functional.cross_entropy(logits, target)
torch.testing.assert_close(loss.detach(), torch.tensor(fixture["hard_loss"]), atol=2e-5, rtol=3e-4)
loss.backward()
maximum = 0.0
for name, p in student.named_parameters():
    expected = torch.tensor(fixture["gradients"][name]).reshape(p.shape)
    assert p.grad is not None, name
    torch.testing.assert_close(p.grad, expected, atol=3e-5, rtol=3e-3, msg=name)
    maximum = max(maximum, (p.grad - expected).abs().max().item())
report = {"ntp_logits_match": True, "branched_mtp_logits_match": True, "hard_loss_match": True,
          "all_gradients_match": True, "max_gradient_absolute_error": maximum}
(root / "comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
