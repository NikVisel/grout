# Packed Q4 Qwen3 / Qwen3.5 implementation

Implemented locally on 2026-09-22. Grout now runs the supplied Qwen3-4B MTP and
MiMo-V2.6-Distill-Qwen-9B text weights with packed 4-bit weights on disk and in
GPU memory. The original FP16/cuTile Qwen3 backend remains available. The packed
backend uses native Rust orchestration and CUDA kernels compiled by NVRTC; it
does not execute Python or downloaded model code during inference.

## Run the supplied models

Requires an NVIDIA GPU with compute capability 8.0 or newer, a CUDA toolkit
with NVRTC, and Grout's existing Rust/cuTile build dependencies. Validated on
Windows with CUDA 13.4 and an RTX 3070 (8 GiB).

The converted directories already exist locally:

```powershell
cargo run --release --bin grout -- --model vendor/models/qwen3-4b-MTP-Q4 --prompt "What is 2 + 2?" --max-new-tokens 32 --max-seq-len 256 --do-mtp --k-toks 3 --strategy conf_adapt --confidence-threshold 0.9
cargo run --release --bin grout -- --model vendor/models/MiMo-Q4 --prompt "What is 2 + 2?" --max-new-tokens 32 --max-seq-len 256
```

To reproduce preparation/conversion, choose output directories that do not exist:

```powershell
python scripts/prepare_mimo_model.py
cargo run --release --bin grout_quantize -- --model vendor/models/qwen3-4b-Inst-2507-MTP --output vendor/models/qwen3-4b-MTP-Q4 --group-size 64
cargo run --release --bin grout_quantize -- --model vendor/models/MiMo-V2.6-Distill-Qwen-9B --output vendor/models/MiMo-Q4 --group-size 64
```

The preparer downloads only missing metadata from pinned MiMo revision
`2367e865d009c13ac81713a2878291d33ab28177`, checking shard extents and tensor/index
membership. It does not overwrite weights or perform a full payload hash check.
The converter refuses an existing output directory and publishes its manifest
only after the conversion finishes. The BF16 source directories are preserved.

| Model | Source text weights | Resident packed weights, including scales |
| --- | ---: | ---: |
| Qwen3-4B MTP | 7.49 GiB | 2.342 GiB |
| MiMo / Qwen3.5-9B text | 16.68 GiB | 5.215 GiB |

These are weights only. KV/recurrent state, activations, CUDA allocations, and
display memory are additional. Load one model at a time on the 8 GiB GPU;
simultaneous residency is not promised. MiMo's vision weights remain in the
source checkpoint and are excluded from this text-only conversion.

With `--max-seq-len 256`, the real-checkpoint smoke suite sampled total GPU
usage of 3364 MiB for Qwen3 and 6308 MiB for MiMo, including a 763 MiB desktop
baseline. These are sampled system-wide measurements, not isolated allocation
or worst-case context peaks.

## Format and computation

`grout-affine-q4-v1` is a Grout-specific weight-only format, not AWQ, GPTQ, NF4,
FP4, or NVFP4. Two unsigned integer codes occupy one `u8`, low nibble first.
Each row/group stores FP32 scale and offset. Groups of 64 use 0.625 bytes per
weight including those parameters. This uses byte storage supported by the
vendor crate and works without native FP4/FP8 Tensor Core instructions.

Large two-dimensional weights, including embedding and output matrices, are
quantized. Norms, convolution kernels, and other small/non-matrix parameters
remain FP32. Qwen3 tied embedding/output weights share the same allocation.
The converter processes one source tensor at a time and writes a separate
safetensors file per tensor with an explicit logical-shape manifest.

Weights remain packed in VRAM; CUDA kernels unpack individual values into
registers. There is no full-model FP16 expansion. Activations and KV cache are
BF16; matrix reductions, softmax reductions, and DeltaNet recurrent state use
FP32. The current matrix kernels use ordinary CUDA arithmetic, not a native
low-bit Tensor Core MMA. Quantization uses round-to-nearest affine groups
without a calibration dataset; task-quality checks are still needed for a
particular deployment.

## Architecture and features

| Component | Implementation |
| --- | --- |
| Format/converter | `src/quantization.rs`, `src/bin/grout_quantize.rs` |
| Architecture/config dispatch | `src/engine.rs`, `src/quantized_config.rs` |
| GPU text inference, cache state, generation | `src/quantized.rs`, `src/quantized.cu` |
| Native Qwen3.5 training | `src/training_qwen35.rs` |
| Shared training objectives/optimizer dispatch | `src/training.rs`, `src/bin/grout_train.rs` |

The Qwen3.5 backend implements mixed Gated DeltaNet/full-attention layers,
depthwise causal convolution, Q/K L2 normalization, learned decay/update gates,
FP32 recurrent matrices, gated RMSNorm, doubled gated query projection, partial
RoPE, zero-centered norms, and untied output embeddings. It accepts both the
nested MiMo configuration and text-only Qwen3.5 configurations. It does not
implement Qwen3.5 MoE, image/video input, or the vision encoder.

The packed backend supports multi-row projections/logits, static MTP,
ConfAdapt, exact output budgets, EOS termination, JSON token/chunk output, and
ordinary greedy or temperature/top-k/top-p sampling. MTP remains greedy.
For rollback, full-attention caches retain a logical prefix while convolution
and recurrent states are explicitly snapshotted and restored. Accepted
predictions are processed as real inputs on the next pass.

The supplied MiMo model has no supported mask-based MTP training metadata;
use ordinary generation for it. Supplying an arbitrary `--mask-id` does not
enable MTP. Native exports with a completed hard/soft MTP `training_run.json`
carry their mask ID through quantization and can use the MTP generator. This
supports trained Qwen3.5 students; it does not establish that the original MiMo
checkpoint learned this objective or that MTP will improve its quality/speed.

The current packed attention kernel supports `--max-seq-len` up to 8192, with
prefill processed in chunks of 32. It does not claim the checkpoint's full 262K
context capability. The packed backend uses eager launches and host logit
readout; CUDA graphs, device argmax, and per-kernel profiling remain features
of the original FP16 backend. `--profile` on packed models reports the weight
footprint and execution precisions instead of per-kernel timings.

Single-user text prompts follow the supplied Qwen3/MiMo templates. MiMo uses
disabled thinking by default; `--enable-thinking` omits the empty thinking block.
Use `--raw-prompt` for externally formatted conversations or tool prompts.
This is not a general Jinja/tool/media renderer, and arbitrary third-party
templates are not supported automatically.
An unrecognized Qwen3.5 template requires `--raw-prompt` instead of silently
applying MiMo formatting to another checkpoint.

## Training and export

The native trainer selects Qwen3 or Qwen3.5 from the source config. It preserves
full-parameter CPU FP32 training, frozen-teacher hard/soft supervision, NTP,
SGD/AdamW, and checkpoint export. Quantized training inputs are rejected: train
from floating-point weights, then quantize the export. No adapter-only training
or low-bit optimizer is silently substituted.

For Qwen3.5 MTP, each token inherits a differentiable recurrent/convolution
history from its last visible predecessor. Real-token history bypasses mask
branches. Full-attention layers retain the blocked visibility and explicit
logical positions. Unsupported noncausal or non-branching recurrent masks are
rejected. Teacher and student token IDs must agree; the original Qwen3-4B and
MiMo vocabularies cannot directly share tokenwise distillation targets.

Small full-parameter training smoke:

```powershell
cargo run --release --features native-training --bin grout_train -- --smoke-test --smoke-architecture qwen3.5 --steps 6 --k-min 2 --k-max 4 --learning-rate 0.003 --output target/qwen35-trained-smoke
```

Choose a new output path when rerunning. Actual training uses `--student`,
`--teacher`, and `--data` as before. `--objective ntp` does not require a mask
token. An MTP run needs a suitable mask ID and matching teacher/student token
semantics. Exports include text weights only; quantization copies the training
metadata and selects the recorded mask for inference.

The small Qwen3.5 run reduced loss from 3.284905 to 2.677223 in six steps.
This is a mechanics check, not model-quality training. Full 9B training was not
run: the FP32 student and teacher text weights alone need about 66.7 GiB,
before gradients, optimizer state, and activations. The current 64 GiB machine
cannot hold that full training configuration.

## Validation

```powershell
cargo test --features native-training --lib
target/qwen35-reference-venv/Scripts/python tests/reference_quantized.py
cargo test --features native-training --lib export_qwen35_training_oracle -- --ignored --nocapture
target/qwen35-reference-venv/Scripts/python tests/reference_training_qwen35.py
python tests/smoke_quantized.py
```

The independent oracle environment is under `target`, with
`transformers==5.12.1` and `safetensors`. It is only for tests. The export test
refuses an existing oracle checkpoint directory; set `GROUT_QWEN35_ORACLE` to a
fresh location for a new Rust fixture (and adjust the Python script path).

To prepare it on a machine with PyTorch already installed:

```powershell
python -m venv --system-site-packages target/qwen35-reference-venv
target/qwen35-reference-venv/Scripts/python -m pip install transformers==5.12.1 safetensors==0.8.0
```

Validated evidence:

- Packed nibble order, constant groups, odd column tails, and bounded rounding
  error; invalid group sizes/non-finite inputs are rejected.
- Independent HF/PyTorch tiny-model BF16 logits: maximum absolute error 0.00390625
  for Qwen3 and 0.0029296875 for Qwen3.5; greedy predictions match.
- Full-prefix, chunked, and temporary-token rollback logits match exactly for
  both architectures in the GPU fixtures.
- Qwen3.5 real-token states are unchanged by inserted/modified mask branches.
- Independent Qwen3.5 CPU NTP and branched MTP logits, hard loss, and every
  parameter gradient match PyTorch; maximum gradient error was approximately
  `3.58e-7`. The teacher has no trainable variables.
- The Qwen3.5 export reloads without changing logits. Existing Qwen3 training
  and MTP unit tests remain in the shared suite.
- A trained tiny Qwen3.5 export was converted to Q4 and exercised through the
  real CLI with its recorded mask ID. ConfAdapt rejection matched ordinary
  decoding. The untrained MiMo checkpoint rejects attempts to force MTP.
- The original FP16 Qwen3 cached-MTP/full-prefix/NTP GPU regression passed.
- An independent full-size MiMo check unpacked the Q4 checkpoint into a BF16
  CPU Transformers model and compared its final prompt logits to the native GPU
  backend. The greedy token matched; cosine similarity was 0.999628, mean
  absolute logit difference 0.04985, and maximum difference 0.390625. This is a
  same-quantized-weights implementation check on one prompt, not a comparison
  against the original BF16 checkpoint's quality. Reproduce with
  `tests/reference_real_quantized.py` in the oracle environment (about 20-25 GiB
  available host RAM required).

Real checkpoint smoke results and sampled system-wide VRAM measurements are
written to `target/quantized-smoke/report.json`. These are narrow functional
checks, not a benchmark, full calibration, or an accuracy guarantee.

## Repository state

No remote changes, commits, pushes, sync, or history rewrites are part of this
implementation. New derived models remain ignored locally. Previously tracked
source shards remain in Git history; `.gitignore` does not remove those blobs.
Resolve that separately before publishing to `https://github.com/NikVisel/grout.git`.
