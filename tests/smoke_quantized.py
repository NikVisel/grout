"""Optional real-checkpoint generation and system-wide GPU-memory smoke tests.

Requires the two converted checkpoints in vendor/models. This is not a quality
benchmark. nvidia-smi memory includes display/other processes, not just Grout.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("target/quantized-smoke")
ROOT.mkdir(parents=True, exist_ok=True)
EXE = Path("target/release/grout.exe")


def used_mib():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(output.splitlines()[0])


def generate(label, model, prompt, flags=(), budget=32):
    output = ROOT / (label + ".json")
    baseline = used_mib()
    peak = baseline
    with (ROOT / (label + ".log")).open("w") as log:
        process = subprocess.Popen([str(EXE), "--model", model, "--prompt", prompt,
            "--max-new-tokens", str(budget), "--max-seq-len", "256", "--warmup-reps", "0",
            "--output-json", str(output), *flags], stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            peak = max(peak, used_mib())
            time.sleep(0.15)
        if process.returncode:
            raise RuntimeError((ROOT / (label + ".log")).read_text(encoding="utf-8"))
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["generated_tokens"] == len(result["token_ids"]) <= budget
    assert result["generated_tokens"] > 0
    if result["mtp_chunks"]:
        assert sum(result["mtp_chunks"]) == result["generated_tokens"]
    return result, {"case": label, "baseline_gpu_mib": baseline, "peak_gpu_mib": peak,
                    "text": result["text"], "tokens": result["generated_tokens"]}


reports = []
qwen = "vendor/models/qwen3-4b-MTP-Q4"
mimo = "vendor/models/MiMo-Q4"
prompt = "What is 2 + 2?"
ntp, report = generate("qwen3-ntp", qwen, prompt)
reports.append(report)
k1, report = generate("qwen3-k1", qwen, prompt, ["--do-mtp", "--k-toks", "1"])
reports.append(report)
assert ntp["token_ids"] == k1["token_ids"], "MTP k=1 differs from NTP"
fallback, report = generate("qwen3-fallback", qwen, prompt,
    ["--do-mtp", "--k-toks", "4", "--strategy", "conf_adapt", "--confidence-threshold", "1"])
reports.append(report)
assert ntp["token_ids"] == fallback["token_ids"], "rejected suffix contaminated subsequent steps"
# Confidence equality is accepted; FP32 softmax can round to exactly one.
# Require actual rejected suffixes, without incorrectly rejecting equal scores.
assert 1 in fallback["mtp_chunks"]
static, report = generate("qwen3-static", qwen, prompt, ["--do-mtp", "--k-toks", "3"], budget=8)
reports.append(report)
assert static["mtp_chunks"] == [3, 3, 2], "static chunks failed the exact output budget"
adaptive, report = generate("qwen3-adaptive", qwen, prompt,
    ["--do-mtp", "--k-toks", "3", "--strategy", "conf_adapt", "--confidence-threshold", "0.9"])
reports.append(report)
assert "4" in adaptive["text"]
for label, prompt, expected in [
    ("mimo-math", "What is 2 + 2?", "4"),
    ("mimo-capital", "What is the capital of France? Answer with one word.", "Paris"),
    ("mimo-code", "Write a Python function named add that returns the sum of two arguments. Only give the code.", "return"),
]:
    result, report = generate(label, mimo, prompt, budget=64)
    reports.append(report)
    assert expected in result["text"], result["text"]
(ROOT / "report.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
