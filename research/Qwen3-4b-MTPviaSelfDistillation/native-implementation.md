# Native Qwen3 MTP implementation

Implemented against the local checkpoint and the authors' published code on 2026-09-22. The original requirements notes in `mtpviaselfdistillation.md` are preserved. Native inference uses the existing Rust/cuTile/cuBLAS engine. Native training uses Rust/Candle autograd on CPU; Python is used only by optional independent validation scripts.

## Checkpoint audit

The supplied directory initially contained three BF16 weight shards and a short README. Config, tokenizer, generation config, and the safetensors index were missing. These have been fetched at Hugging Face revision `5273e79569b30ca334fe4093d32a225b327fde0a`. `scripts/prepare_mtp_model.py` can repeat the preparation without overwriting existing files; it checks shard sizes, headers, and all 398 indexed tensor names. This is not a complete file-hash verification.

The actual checkpoint has 36 layers, hidden size 2560, intermediate size 9728, 32 query heads, 8 KV heads, head dimension 128, and vocabulary size 151936. It uses tied input/output embeddings; there is no separate MTP head and no embedding resize is required. The tokenizer maps `<|mtp_special_token_0|>` to 151669. The generation config already specifies both EOS IDs, 151645 and 151643. The existing generation-config deserializer and engine already supported those multiple IDs.

The checkpoint's single-user chat template ends at `<|im_start|>assistant\n`. Grout's previous unconditional empty thinking block is omitted for this MTP tokenizer. `--raw-prompt` continues to allow exact external formatting. This implementation does not add a general Jinja conversation/tool template renderer.

## Paper-to-code map

The [paper's methodology and implementation sections](https://arxiv.org/html/2602.06019v2#S3) motivate both inference and training. Their executable details are checked against the [checkpoint generation code](https://huggingface.co/jwkirchenbauer/Qwen3-4B-Inst-2507-MTP/blob/5273e79569b30ca334fe4093d32a225b327fde0a/modeling_qwen3.py) and the [authors' training code](https://github.com/jwkirchenbauer/mtp-lm/blob/167413ea3c0113a51c6f7f3f281f60324169c608/litgpt/pretrain.py).

| Section | Native implementation | Files and entry points |
| --- | --- | --- |
| 3.1 NTP | Existing single-token inference retained; differentiable shifted-token cross entropy added | `src/model.rs::generate`; `src/training.rs::ntp_loss` |
| 3.2 MTP | Insert `k-1` masks; read the final `k` positions through the shared LM head | `src/model.rs::generate_mtp`, `build_step_graph_rows`; `src/mtp.rs::MtpBatch` |
| 3.3 Student-forced training | Greedy student proposals condition a frozen teacher; hard-teacher CE or soft-teacher KL; full student gradients and optimizer updates | `src/training.rs::student_forced_loss`, `distillation_loss`, `NativeOptimizer`; `src/bin/grout_train.rs` |
| 3.4 Inference | Deterministic parallel readout; no verifier model or speculative acceptance stage | `src/model.rs::generate_mtp`; `src/mtp.rs::top1` |
| 4.1 Tokenization/masking | Ground-truth tokens retained between inserted regions; explicit original logical positions; randomized span size and placement | `src/mtp.rs::MtpBatch`; training CLI sampler; `NativeQwen3::forward` |
| 4.2 Blocked attention | Ground-truth queries skip mask regions; mask queries see the earlier ground truth and their own causal region | `MtpBatch::visible`, `attention_bias`; differentiable attention in `src/training.rs` |
| 4.3 Static/ConfAdapt | Fixed-k or longest contiguous prefix above threshold, with at least one token per pass | `src/mtp.rs::accepted_count`; CLI flags in `src/main.rs` |

## Existing engine changes

`src/model.rs` was the main inference integration point. Its original StepGraph supported multiple input rows but always gathered just the final hidden row before a matrix-vector LM projection. `GatherRows` and a matrix-matrix final projection now return the required suffix logits. The MTP path caches StepGraphs and tensor pools by query length and logit-row count within a request. It uses eager execution; the original single-token CUDA graph remains the NTP path.

The existing causal prefill attention already accepts a nonzero query start and reads persistent KV storage. It is reused for MTP suffix attention. The non-fused host KV-write kernel in `src/kernels.rs::kv_cache_update_seq_f16` did assume a zero start. It now translates absolute cache slots to local input rows and handles unaligned nonzero offsets. This also fixes the eager one-token fallback's use of that kernel. The existing fused multi-row Q/K normalization, RoPE, and KV writer can already handle contiguous nonzero starts.

`src/cublas.rs` already supports matrix-matrix projections with variable row counts, so no new GEMM implementation was necessary. `vendor/cuda-core/src/simt/embedded.rs` handles embedded module loading; this MTP implementation needed no changes there.

`src/main.rs` adds `--do-mtp`, `--k-toks`, `--mask-id`, `--strategy static|conf_adapt`, `--confidence-threshold`, and `--output-json`. `GenerationOutput` now exposes token IDs and emitted chunk lengths. MTP rejects sampling, resolves the mask from the tokenizer by default, stops at the first accepted EOS, and reduces k to use the exact remaining token budget.

## Cache invariant and confidence semantics

Let `C` be the number of real input tokens already represented by committed K/V. The first pass processes the prompt followed by `k-1` masks. Following passes process the previous `a` accepted real tokens followed by `k-1` fresh masks, starting at `C`. Their query length is `a+k-1`, which can reach `2k-1`.

After a pass, advance `C` only by the real input rows processed in that pass. All mask K/V is temporary, including positions whose predictions were accepted. Accepted predictions must be embedded and processed on the next pass to obtain their real K/V. The fixed cache allocations do not need physical compaction: subsequent writes overwrite the temporary suffix, and attention's KV length excludes any stale tail. Merely rewinding by `k-a` would retain incorrect mask states.

Confidence is the top token's unscaled softmax probability. The reference rejects a row only when confidence is **less than** the threshold, so equality is accepted. The first low-confidence position cuts off the entire later suffix; if the first position fails, one token is still emitted. The CPU confidence reduction is numerically stabilized with a maximum subtraction. Inference currently transfers the k-row logits to the host; fusing rowwise top-1/confidence on the GPU is an optimization still available.

Normal contiguous RoPE positions are sufficient for this inference layout. Repeated logical position IDs and blocked visibility are needed in the packed training layout and are implemented in the differentiable training model. A new blocked cuTile attention kernel is not required to run this already-trained checkpoint.

## Training implementation and boundaries

The trainer loads standard Qwen3 safetensors into trainable Rust tensors. It implements embeddings, GQA, Q/K RMSNorm, RoPE with explicit positions, blocked attention, SiLU MLPs, residuals, and a tied or untied output head. RMSNorm and RoPE use differentiable primitive tensor operations. Every student parameter is trainable; the separately loaded teacher has no trainable variables.

For each packed example, student argmax proposals replace the mask input slots in a teacher pass. The teacher's next-token distributions supervise the corresponding student rows. Hard supervision uses the teacher's argmax labels, including positions where they disagree with the student's proposal. Soft supervision computes tokenwise teacher-to-student KL. The implementation follows `pt_ce_plus_ent_loss` with entropy coefficient zero in the authors' code; it does not turn Equation 3 into an ad hoc sequence-probability reward on the student's own chosen tokens.

The CLI accepts JSONL text or token-ID documents, preserves document boundaries, appends the tokenizer EOS, and randomizes k and region placement with a seed. It uses one document chunk per optimization step. Input text formatting is the caller's responsibility; automatic chat wrapping is not applied to training text. The final MTP window may extend beyond the document EOS, but it cannot see the next document.

Both SGD and AdamW are implemented through Candle. Exports contain safetensors, an index, config, tokenizer metadata, and training-run metadata; Grout can consume that format. Optimizer state, distributed training, GPU training, gradient accumulation, and exact replication of the authors' data sampler/schedule are not implemented. Reloading an exported student starts a new optimizer. These limits mean this is a native implementation of the requested mechanisms, not reproduction of the paper's large training run.

The current machine has an RTX 3070 with 8 GB VRAM and 64 GB RAM. The supplied 4B checkpoint runs through the cuTile inference path. CPU f32 full-parameter AdamW training needs roughly 80 GB just for student, teacher, gradients, and two optimizer moments, before activations and temporary storage. SGD reduces the persistent model/training state, but a full 4B run still needs substantial RAM. No full 4B retraining result or paper accuracy/speedup claim is implied by the small-model tests.

## Commands

Prepare and validate metadata if necessary:

```powershell
python scripts/prepare_mtp_model.py --reference-code target/mtp-reference
```

Native inference:

```powershell
cargo run --release -- --model vendor/models/qwen3-4b-Inst-2507-MTP --prompt "What is 2 + 2?" --max-new-tokens 32 --max-seq-len 128 --do-mtp --k-toks 16 --strategy conf_adapt --confidence-threshold 0.9
```

Use `--strategy static --k-toks 3` for fixed blocks, or `--k-toks 1` for the MTP loop's NTP baseline. Omit `--do-mtp` to use the original generator. `--output-json target/result.json` saves token IDs and chunk lengths for comparisons. The regular `--max-new-tokens` limit excludes the prompt; the reference's `max_returned_tokens` includes it.

Native training smoke test (choose an output directory that does not exist):

```powershell
cargo run --release --features native-training --bin grout_train -- --smoke-test --output target/mtp-trained-tiny --steps 6 --k-min 2 --k-max 4 --learning-rate 0.003
```

Training an actual checkpoint uses `--student <directory> --teacher <directory> --data <JSONL> --output <new-directory>` instead of `--smoke-test`. The MTP student tokenizer must define the mask token or receive `--mask-id`. Teacher and student token IDs must agree. For paper initialization, prepare the original Qwen3-4B-Instruct-2507 checkpoint with the MTP token metadata for the student and use the frozen original checkpoint as teacher. Loading the supplied MTP weights as the student continues training from that model.

Run deterministic correctness checks:

```powershell
cargo test --features native-training --lib
cargo test --release --lib cached_mtp_matches_full_prefix_and_ntp -- --ignored --nocapture
```

Optional PyTorch oracle setup lives entirely under `target`:

```powershell
python -m venv --system-site-packages target/mtp-reference-venv
target/mtp-reference-venv/Scripts/python -m pip install transformers==4.57.3 safetensors
cargo test --features native-training --lib export_training_oracle -- --ignored
target/mtp-reference-venv/Scripts/python tests/reference_training.py
```

The oracle needs an installed PyTorch; it is not a runtime dependency of either Rust binary. `tests/reference_mtp.py` compares a native `--output-json` artifact with the authors' pinned generation implementation. It normalizes the reference's whole-block EOS behavior to Grout's first-EOS stopping convention. The reference can leave an incomplete final token budget unused; Grout deliberately fills that budget with a smaller final block.

## Validation evidence

- Local checkpoint: 398 indexed BF16 tensors across three complete-size shards; standard Qwen3 tensor shapes and tied vocabulary confirmed.
- Static native k=3: `2 + 2 = 4.`, three emitted chunks of size 3, including EOS in the final chunk.
- Native ConfAdapt k=16, threshold 0.9: `2 + 2 = 4.`, chunk sizes `[4, 5]`; all nine IDs and both chunk sizes match the authors' float32 CPU implementation.
- Training: six native optimization steps completed and exported a tiny checkpoint. Unit tests cover hard/soft loss gradients, NTP training, teacher immutability, packed-vs-independent region logits, confidence boundaries, and exact checkpoint reload.
- Independent Hugging Face/PyTorch training oracle: logits, hard-teacher loss, and all parameter gradients match; maximum absolute gradient difference `3.5762786865234375e-7`.
- The GPU hardware test passed: cached output equals full-prefix recomputation for static k=2, 3, 16 and ConfAdapt; k=1 equals NTP. It also passed zero-token and partial final-block checks on the supplied 4B checkpoint. The test is explicitly ignored by default to avoid loading 4B weights during ordinary unit tests.

MTP CUDA graph capture, GPU confidence reduction, and performance benchmarking are subsequent optimization work. Average emitted tokens per pass is recorded separately from elapsed time and must not be reported as measured throughput speedup.
