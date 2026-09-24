# MiMo / Qwen 3.5 compatibility and quantization audit

This document records the pre-implementation baseline. Subsequent implementation
and validation are documented in [the implementation guide](quantized-qwen35-implementation.md).

Audited 2026-09-22 against feature commit `235ab5f` and HEAD `c48b970`.
This is an implementation assessment, not an implementation of Qwen 3.5 or
quantized inference. Here, “both models” means the local Qwen3-4B MTP checkpoint
and MiMo-V2.6-Distill-Qwen-9B. The proposed Qwen 3.5 backend also covers the
original dense Qwen3.5-9B architecture.

## Current result

**MiMo cannot run in the current Grout engine.** Adding its missing metadata is
necessary but insufficient. It requires a different decoder architecture.
Neither local checkpoint currently uses 4-bit or 8-bit quantized weights.

| Checkpoint | Local storage | Grout inference | Native training |
| --- | --- | --- | --- |
| Qwen3-4B-Inst-2507-MTP | BF16, 398 tensors | FP16 weights; existing Qwen3 MTP path works | CPU FP32 |
| MiMo-V2.6-Distill-Qwen-9B | BF16, 760 tensors | Unsupported | Unsupported |

Evidence collected:

- Read all seven local safetensors headers without loading the weight payloads.
  Every shard's file length equals the header's declared payload extent.
- MiMo's 760 tensor names and shard assignments exactly match the official index
  at revision `2367e865d009c13ac81713a2878291d33ab28177`. This checks structure,
  not payload hashes or numerical integrity.
- MiMo's directory contains only four shards and a README. The current release
  executable exits with `failed to read .../config.json` before inference.
- A smoke run of the existing release executable on Qwen3-4B with ConfAdapt,
  `k=3`, threshold `0.9`, and an eight-token budget produced `2 + 2 = 4.` with
  chunks `[3, 3, 2]`. This verifies that binary's existing path, not a new build
  or general model quality.
- Official MiMo metadata, header audit results, and smoke output are saved under
  the ignored `target/compatibility-audit/` directory.

The publisher identifies MiMo as a supervised fine-tune of Qwen3.5-9B.
See the [official model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
and [pinned configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/config.json).

## Required architecture work

Current integration points are `src/config.rs::Qwen3Config`,
`src/loader.rs::WeightLoader`, `src/model.rs::Qwen3Engine`, and
`src/training.rs::NativeQwen3`. The engine requires `head_dim == 128`, flat Qwen3
configuration, `model.layers.*` names, and ordinary attention in every layer.

MiMo has a nested `text_config`, 32 text layers, hidden size 4096, MLP size 12288,
vocabulary 248320, and untied embeddings/output projection. Its text weights use
`model.language_model.*`; its image encoder uses `model.visual.*`.

Implement a separate Qwen 3.5 decoder selected by model type, retaining the
existing Qwen3 decoder and its tests:

1. Load nested text/vision configuration and the correct names and shapes.
   Fetch config, index, tokenizer, generation metadata, and chat template from a
   pinned revision. Reject unsupported architectures explicitly.
2. Add the 24 Gated DeltaNet layers: causal depthwise convolution, Q/K L2
   normalization, learned decay/update gates, recurrent updates, gated
   normalization, and separate convolution/recurrent state. Implement prefill,
   single-token decode, and multi-row suffix processing.
3. Adapt the eight full-attention layers for 16 query heads, four KV heads,
   dimension 256, and the doubled query projection containing an output gate.
4. Implement partial RoPE: only 64 of 256 head dimensions rotate. Preserve
   Qwen 3.5's position conventions. Use zero-centered `1 + weight` RMSNorm where
   required; the DeltaNet gated norm has its own semantics.
5. Support the checkpoint's MiMo chat template, thinking controls, and EOS IDs.
   The current hard-coded Qwen chat wrapper is insufficient for exact formatting.
6. Start with explicit text-only support, skipping image-encoder weights. Image
   and video input additionally require the vision encoder, preprocessing,
   multimodal position handling, and embedding insertion.

These operations are specified in the
[Transformers Qwen 3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The original [Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
uses the same dense hybrid architecture. This assessment does not cover the
Qwen 3.5 MoE variants.

## Preserving the recent MTP and training features

Keep multi-position logits, static MTP, ConfAdapt, exact token budgets, EOS
handling, output token/chunk reporting, and the existing Qwen3 CUDA-graph path.
MTP currently runs eagerly; the recent commit did not implement MTP CUDA graphs.

**MiMo is not established as a checkpoint for the current mask-based MTP
algorithm.** Its config includes `mtp_num_hidden_layers: 1`, but the local
weights have no named MTP module and the official tokenizer has no
`<|mtp_special_token_0|>`. The publisher describes SFT, not the recent commit's
self-distilled mask objective. A config field, a manually supplied mask ID, or
the word “Distill” does not make that algorithm work. Preserve MTP for the
trained Qwen3 checkpoint; require appropriate training before enabling it for
MiMo. A separate native draft-head/speculative MTP implementation would be a
different algorithm and would need its corresponding weights and verification.

The following are design requirements inferred from Grout's mask layout and
Qwen 3.5's recurrent updates:

- **Inference:** process real inputs, then snapshot the convolution and recurrent
  states at that boundary. Process masks on temporary states and discard those
  states afterward. On the next pass, feed accepted predictions as real tokens.
  For full-attention layers, keep Grout's existing committed-KV-prefix invariant.
  Changing a cache length alone cannot undo a DeltaNet state update.
- **Training:** the current packed layout lets real tokens skip mask regions.
  A recurrent layer cannot implement that by receiving a dense attention bias.
  Maintain a real-token state stream and differentiable branches for mask
  regions; discard a branch before continuing the real stream. Convolution
  histories need the same isolation. Full-attention layers retain blocked
  visibility and explicit logical positions.
- **Trainer:** add differentiable Qwen 3.5 operations and architecture dispatch
  while retaining frozen-teacher supervision, hard/soft losses, optimizers,
  and export. Do not silently change the current full-parameter training mode
  into adapter-only training.
- **Teacher compatibility:** Qwen3-4B and MiMo have different token vocabularies.
  They cannot directly share the current tokenwise KL/CE distillation path.
  Use matching token semantics for teacher/student, or design a separate
  cross-tokenizer distillation method. Equal vocabulary size alone is not an
  adequate compatibility check.

## Quantized storage and execution

`WeightLoader` converts supported floating-point inputs into `Vec<f16>` and
`Tensor<f16>`. Other dtypes are rejected. Model projections and `src/cublas.rs`
use FP16 operands. The native trainer loads FP32 and exports its FP32 variables.
No CLI dtype setting or end-to-end quantized model path exists.

The vendor types provide useful building blocks:

| Type | Meaning and proposed use |
| --- | --- |
| `i8`, `u8` | INT8 values, or bytes holding packed integer 4-bit values |
| `f8e4m3fn`, `f8e5m2` | FP8 storage; require scales and supported execution kernels |
| `f4e2m1fnx2` | Two FP4 E2M1 values in one byte; low nibble first |
| `f4e2m1fn`, `i4` | Logical sub-byte markers, not standalone byte-addressable host `DType` tensors |
| `f8e8m0fnu` | Exponent-only scale format, not a general signed weight format |

The pinned cuTile checkout already contains FP4 pack/unpack, `mmaf_scaled`,
and NVFP4/MXFP8 examples. Thus the gap is not solely compiler type recognition:
Grout still needs quantization, checkpoint serialization, model dispatch, and
kernels appropriate for the actual GPU.

The machine reports **RTX 3070, 8192 MiB, compute capability 8.6**. It has no native
FP8/FP4 Tensor Core arithmetic. NVIDIA documents Ampere's supported precisions
in its [GA10x architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf).
Native NVFP4 targets Blackwell; the pinned cuTile example explicitly skips
unsupported GPUs. FP4 storage can still be used on this GPU through software
unpacking/dequantization followed by supported arithmetic.

For this machine, implement **4-bit weights with 16-bit activations** for MiMo.
Use the same path for Qwen3-4B, with INT8 as a less aggressive alternative.
Packed integer weights in `u8` are a practical first format; if using the vendor's
`f4e2m1fnx2`, implement and evaluate an E2M1 weight-only format explicitly.
It is not interchangeable with integer INT4, NF4, or a complete NVFP4 recipe.

Keep weights compressed on disk and in VRAM. Unpack only tiles inside a fused
kernel, or use bounded reusable scratch for an initial correctness path.
Expanding the whole model into FP16 at load time would defeat the runtime memory
requirement. Accumulation, normalization, softmax, and recurrent state should
retain higher precision; use FP32 recurrent state initially. Evaluate FP16
versus BF16 activations for Qwen 3.5's range before choosing the production path.

Required quantization work:

1. A versioned weight representation recording logical/packed shape, format,
   group size, scale dtype/layout, and zero points when applicable. Raw `u8`
   safetensors plus explicit metadata can represent packed weights without
   pretending each byte is one logical model element.
2. A streaming converter with calibration/error measurement and reload checks.
   Preserve the BF16 sources; write separate derived checkpoints.
3. Quantized GEMV for decode and GEMM for prefill **and multi-row MTP**. Support
   fused QKV/gate-up layouts, output projections, and quantized embedding lookup.
   Preserve shared storage for Qwen3's tied embedding/output weights.
4. Hardware-aware dispatch and graph-safe stable buffers. Avoid persistent
   full-size FP16 copies or duplicated packed/fused matrices.
5. Quantized export after training. Full-parameter optimization still needs
   higher-precision trainable state. A frozen quantized base with trainable
   adapters is an optional additional training mode, not already implemented.

For future native NVFP4, use the correct scaling recipe: E2M1 payloads, E4M3
block scales, and an additional FP32 tensor scale. E8M0 is associated with
microscaling schemes and is not an automatic substitute. See
[NVIDIA's format definitions](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html).

## Memory budget

Calculated from local tensor shapes. These are ideal weight-payload sizes in
GiB; quantization scales, higher-precision exceptions, caches, scratch,
activations, driver/display allocations, and loading peaks are additional.

| Weights | BF16/FP16 | 8-bit | 4-bit |
| --- | ---: | ---: | ---: |
| Qwen3-4B MTP | 7.49 | 3.75 | 1.87 |
| MiMo text only | 16.68 | 8.34 | 4.17 |
| MiMo vision encoder, additional | 0.85 | 0.42 | 0.21 |

MiMo text weights alone exceed this GPU's capacity at 8-bit. Its separate
embedding and output matrices consume **3.79 GiB** if both stay 16-bit, so they
must be considered in the quantization budget. Two 4-bit models have an ideal
combined payload of about **6.04 GiB** before overhead; simultaneous residency
must be measured and cannot be promised. Loading them one at a time is the
initial feasible target. Context length needs an explicit memory budget too.

## Implementation order and acceptance evidence

1. Preserve the current Qwen3 baseline and implement reusable quantized weights
   and kernels there first. Verify disk reload and actual resident allocations.
2. Add Qwen 3.5 text inference with a small differentiable/reference model for
   correctness, then load the quantized real MiMo checkpoint within the budget.
3. Verify full-prefix versus cached logits, chunked versus unchunked prefill,
   mixed attention types, partial RoPE, gates, and both recurrent state buffers
   against a pinned independent implementation.
4. Retain Qwen3 MTP regression tests: `k=1` versus NTP in the same dtype,
   multi-row logits, static/ConfAdapt, rejected suffixes, EOS, and token limits.
   For Qwen 3.5 training, test branch isolation, losses, gradients, and export
   before claiming the MTP objective is supported.
5. Measure quantization quality, finite logits, task accuracy, ConfAdapt chunk
   distributions, peak VRAM, and throughput. Quantization can alter confidence
   and accepted chunks; exact agreement with BF16 is not an acceptance rule.

No Qwen 3.5 inference, quantized execution, or quantized-training success is
claimed by this audit. Those are implementation work, not configuration switches.

## Git state

No fetch, pull, push, sync, remote change, or history rewrite was performed.
`origin` still points at `https://github.com/huggingface/grout.git`; the intended
fork is `https://github.com/NikVisel/grout.git`.

There is also a concrete issue to resolve before eventual publishing:
`git ls-files vendor/models` still lists 18 tracked files, including weight
shards. The later `.gitignore` change does not remove files already tracked or
remove their blobs from commit history. Changing the remote alone will not
address that. History and tracked weights were left untouched during this audit.
