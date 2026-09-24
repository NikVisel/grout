User Question: 
Implementation update: [Native implementation, code map, commands, and validation](native-implementation.md).

Can you identify the parts of this codebase that would need to be modifed to support https://huggingface.co/jwkirchenbauer/Qwen3-4B-Inst-2507-MTP ; We would want to actually implement section 3 (3.1, 3.2, 3.3, 3.4) and section 4 (4.1, 4.2, 4.3) https://arxiv.org/html/2602.06019v2 ;



GPT Luna Response: 

...

Yes. The repository currently implements standard single-token Qwen3 inference; supporting the MTP checkpoint and paper sections requires work in five areas.

**Current architecture**

- `model.rs:1276` owns the complete inference engine.
- `model.rs:1858` has a strictly one-token-at-a-time `generate()` loop.
- `model.rs:495` allocates decode buffers with sequence length fixed to `1`.
- `model.rs:320` captures a CUDA graph assuming one query token per decode step.
- `kernels.rs:617` has dynamic-position RoPE, but only supports the current sequential layout.
- `kernels.rs:780` has cache update kernels, but the decode variant assumes `seq_len == 1`.
- `config.rs:5` supports only the existing Qwen3 configuration fields and a single configured EOS token.

## Required changes by paper section

### Sections 3.1 and 3.2: NTP and MTP outputs

The existing NTP path is reusable, but its interfaces need to become multi-position aware.

Modify:

- `model.rs:1858`
  - Add static `k` generation.
  - Append `k - 1` mask tokens to the current prefix.
  - Run one forward pass producing `k` logits.
  - Return or materialize logits for every predicted position, rather than only the final position.
- `model.rs:495`
  - Replace decode-only `[1, ...]` buffers with buffers sized for a configurable query length.
  - Likely support query length up to `2k - 1`, as described in the paper’s KV-cache procedure.
- `model.rs:5060`
  - Generalize RoPE position handling so each query row can receive an explicit position ID.
  - MTP positions must use the original logical positions, not merely their physical location in the expanded input.
- `kernels.rs:617`
  - Add a position-ID tensor/vector variant of RoPE, rather than only `position_start + seq_idx`.
- LM-head and argmax code in `model.rs:1858`
  - Produce `k` token IDs and confidence values.
  - Confidence is needed by ConfAdapt in section 4.3.

### Section 3.3: student-forced online MTP training

This is the largest missing subsystem. The current Rust project has no optimizer, autograd, dataset pipeline, teacher model, or training executable.

You would need to add:

- A training executable, probably under `src/bin/` or a separate training crate.
- Dataset/tokenization preparation.
- A frozen NTP teacher model.
- A trainable MTP student model initialized from the teacher checkpoint.
- Student argmax rollout.
- Teacher-forced scoring of the student-generated tokens.
- The loss from Equation 3:

  $$
  \mathcal{L}_{MTP}
  =
  -\frac{1}{R}
  \sum_r
  P_{\theta_T}(y'_r \mid x_{1:i_r})
  \log P_{\theta_S}(y'_r \mid x_{1:i_r})
  $$

- Randomized `k`, typically `[2, 16]`.
- Hard-teacher argmax supervision for the paper’s main configuration.
- Frozen teacher parameters and trainable student parameters.
- Optimizer and checkpoint export.

This cannot be implemented by modifying only the current inference code. The practical approach is to use the authors’ PyTorch/LitGPT training implementation and use this Rust engine for inference, or add a separate PyTorch-based training component to this repository.

The authors’ reference implementation confirms the relevant locations conceptually: `litgpt/pretrain.py`, `litgpt/model.py`, and `litgpt/mtp.py`.

### Section 4.1: tokenization and masking

Add a training-side batch preparation module.

It must:

- Add or recognize the `<MTP>` token.
- Insert `k - 1` MTP mask tokens after each selected prefix.
- Replicate the ground-truth continuation tokens after each MTP region.
- Preserve original logical position IDs despite physical replication.
- Support multiple MTP regions in one sequence.
- Randomize region offsets and `k`.
- Handle EOS by padding the final MTP region with EOS tokens.

Relevant existing areas:

- [src/config.rs](src/config.rs#L5): add mask-token configuration and multiple EOS IDs.
- [src/model.rs](src/model.rs#L1858): inference-side mask-token handling.
- [src/model.rs](src/model.rs#L5060): position-aware RoPE.
- New code: a dedicated `mtp_batch.rs` or equivalent training/data module.

The Hugging Face model card specifies `mask_id=151669` for this checkpoint and may use both `151645` and `151643` as stop tokens. These should be read from tokenizer/config metadata rather than hardcoded.

### Section 4.2: blocked attention

The current attention kernels implement ordinary causal attention only. For training, add a blocked causal attention mask with these rules:

- Ground-truth tokens attend to upstream ground-truth tokens.
- Ground-truth tokens skip MTP-mask spans.
- MTP tokens attend to upstream ground-truth tokens.
- MTP tokens attend causally within their own MTP span.
- MTP tokens cannot attend to other MTP spans.
- Multiple spans can coexist in one sequence.
- Offset and span length must be dynamic.

Likely changes:

- [src/kernels.rs](src/kernels.rs#L1200): add a blocked-mask attention kernel or an attention-mask tensor input.
- Existing prefill attention call sites in [src/model.rs](src/model.rs#L3060): pass mask metadata or use a new blocked-attention path.
- [src/model.rs](src/model.rs#L3420): generalize prefill position handling and attention invocation.

The existing `fmha_prefill_causal` and `fmha_prefill_gqa` kernels are useful starting points, but their mask logic currently assumes:

```text
key_position <= query_position
```

They need a per-query/per-key visibility predicate instead.

### Section 4.3: static and adaptive decoding

Modify [src/model.rs](src/model.rs#L1858) and the CLI in [src/main.rs](src/main.rs#L1):

Add options equivalent to:

- `--do-mtp`
- `--k-toks`
- `--mask-id`
- multiple `--eos-id` values
- `--strategy static`
- `--strategy conf_adapt`
- `--confidence-threshold`

For ConfAdapt:

1. Run the model with `k_max` positions.
2. Compute the top-token probability at every predicted position.
3. Select the largest prefix length `k'` where every confidence exceeds $\tau$.
4. Commit only those `k'` tokens.
5. Remove unused mask-token KV entries.
6. Continue generation from the committed prefix.

The paper’s rule is effectively:

$$
k'
=
\max
\left\{
j:
\max_v P(x_{i+j}=v) > \tau
\text{ for all } j \leq k'
\right\}.
$$

## KV-cache work required for real speedup

This is not explicitly numbered in sections 3.1-4.3, but it is necessary to make the implementation functional.

The paper’s cache procedure requires:

- Generate `k` predictions using `k - 1` mask slots.
- Remove mask-token KV entries.
- Commit the accepted/generated token KV entries.
- Append the accepted tokens plus new mask tokens for the next step.
- Reuse ordinary causal attention during inference.

Current limitations:

- `model.rs:320` captures a one-token decode graph.
- `model.rs:495` has one-token buffers.
- `kernels.rs:780` updates one decode position.
- `model.rs:1858` advances `cur_pos` by exactly one.
- The KV cache is currently append-only and has no pop/compact operation.

You will need either:

1. A new variable-query-length decode path, probably initially without CUDA graphs; or
2. A generalized CUDA graph family keyed by query length, with graphs for lengths from `k` through `2k - 1`.

The first option is much easier to validate. CUDA graph optimization should come afterward.

## Model loading changes

The checkpoint is still structurally Qwen3, so the existing weight loader should mostly work. However:

- `loader.rs:1` must tolerate the MTP-expanded embedding/lm-head vocabulary if the checkpoint contains an added mask token.
- `config.rs:5` should support array-valued EOS IDs.
- Tokenizer metadata should identify the mask token and stop tokens.
- The loader should validate that the configured `mask_id` is within the embedding vocabulary.
- The current `tie_word_embeddings` handling should be checked against the MTP checkpoint’s actual tensor shapes.

## Tests to add

The existing tests only cover standard generation and individual kernels. Add:

- Mask-token tokenizer ID validation.
- Multiple EOS-token handling.
- Position-ID/RoPE tests with replicated physical positions.
- Blocked-attention visibility tests.
- `k=1` equivalence with current NTP generation.
- Static `k=2`, `k=3`, and `k=16` generation.
- ConfAdapt acceptance tests with synthetic logits.
- KV-cache compaction tests.
- End-to-end MTP generation against the Hugging Face reference implementation.

## Recommended implementation order

1. Add metadata support for mask ID and multiple EOS IDs.
2. Implement a non-graph, host-controlled static MTP decode path.
3. Generalize forward buffers, RoPE, cache updates, and LM-head output to multiple positions.
4. Add KV-cache removal/compaction.
5. Add ConfAdapt.
6. Add blocked attention and MTP batch preparation.
7. Add the student/teacher training pipeline.
8. Add CUDA graph variants and performance tuning.

The main conclusion is that inference support is concentrated in `model.rs`, `kernels.rs`, `config.rs`, and `main.rs`. Implementing the paper’s training method, especially section 3.3, requires a new training stack rather than a small extension to the current Rust inference engine.
