#![allow(clippy::too_many_arguments)]

// Kinds that fire during the normal Qwen3 inference path (decode CUDA graph
// + step-graph prefill) and therefore need warmup to pre-pay JIT cost. The
// `RmsNormPersistent` kernel is excluded — its env-flag dispatch was never
// wired up, so it's dead code in runtime; warming it would just burn JIT
// time. The kernel itself stays in kernels.rs for ablation runs.
//
// Gemm / Gemv run via cuBLAS rather than cutile JIT, but we keep them in
// the list: first-call cuBLAS handle + workspace setup is not free and
// paying it in warmup keeps the first generate() tighter.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelKind {
    EmbeddingBatch,
    Gemm,
    Gemv,
    RmsNorm,
    RopeSeq,
    KvCacheUpdateSeq,
    FlashAttnCausalSeq,
    AddVec,
    SiluMul,
    GatherRow,
    ArgmaxBlocks,
    AddRmsNorm,
    QkNorm,
    QkRope,
    QkNormRopeKvPrefill,
    QkNormRopeKvDecode,
    ArgmaxReduceBlocks,
}

impl KernelKind {
    pub const COUNT: usize = 17;

    pub const fn idx(self) -> usize {
        self as usize
    }
}

pub const TILE_KERNEL_KINDS: [KernelKind; 17] = [
    KernelKind::EmbeddingBatch,
    KernelKind::Gemm,
    KernelKind::Gemv,
    KernelKind::RmsNorm,
    KernelKind::RopeSeq,
    KernelKind::KvCacheUpdateSeq,
    KernelKind::FlashAttnCausalSeq,
    KernelKind::AddVec,
    KernelKind::SiluMul,
    KernelKind::GatherRow,
    KernelKind::ArgmaxBlocks,
    KernelKind::AddRmsNorm,
    KernelKind::QkNorm,
    KernelKind::QkRope,
    KernelKind::QkNormRopeKvPrefill,
    KernelKind::QkNormRopeKvDecode,
    KernelKind::ArgmaxReduceBlocks,
];

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2, max_divisibility=16,),
                       ))]
    fn gemm_f16<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f16, { [BM, BN] }>,
        x: &Tensor<f16, { [-1, K] }>,
        y: &Tensor<f16, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x: Tile<f16, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tile_y: Tile<f16, { [BK, BN] }> = part_y.load([i, pid.1]);
            let tile_x: Tile<f32, { [BM, BK] }> = convert_tile(tile_x);
            let tile_y: Tile<f32, { [BK, BN] }> = convert_tile(tile_y);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        let tile_z: Tile<f16, { [BM, BN] }> = convert_tile(tile_z);
        z.store(tile_z);
    }

    unsafe fn load_f16_ptr(
        ptrs: &Tensor<i64, { [-1] }>,
        group_id: i32,
    ) -> PointerTile<*mut f16, { [] }> {
        let one_shape: Shape<{ [1] }> = Shape::<{ [1] }> { dims: &[] };
        let ptr_part: Partition<i64, { [1] }> = ptrs.partition(one_shape);
        let ptr_int: Tile<i64, { [1] }> = ptr_part.load([group_id]);
        let ptr_int: Tile<i64, { [] }> = ptr_int.reshape(const_shape![]);
        let ptr: PointerTile<*mut f16, { [] }> = int_to_ptr(ptr_int);
        unsafe { assume_div_by::<_, 16>(ptr) }
    }

    unsafe fn load_f16_desc_2d(
        ptrs: &Tensor<i64, { [-1] }>,
        metas: &Tensor<i32, { [-1, 8] }>,
        group_id: i32,
    ) -> (Tensor<f16, { [-1, -1] }>, i32, i32) {
        let meta_part: Partition<i32, { [1, 8] }> = metas.partition(const_shape![1, 8]);
        let row: Tile<i32, { [1, 8] }> = meta_part.load([group_id, 0i32]);
        let idx0: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let idx1: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let idx2: Tile<i32, { [] }> = scalar_to_tile(2i32);
        let rows_tile: Tile<i32, { [1, 1] }> = extract(row, [idx0, idx0]);
        let cols_tile: Tile<i32, { [1, 1] }> = extract(row, [idx0, idx1]);
        let stride0_tile: Tile<i32, { [1, 1] }> = extract(row, [idx0, idx2]);
        let rows_tile: Tile<i32, { [] }> = rows_tile.reshape(const_shape![]);
        let cols_tile: Tile<i32, { [] }> = cols_tile.reshape(const_shape![]);
        let stride0_tile: Tile<i32, { [] }> = stride0_tile.reshape(const_shape![]);
        let ptr: PointerTile<*mut f16, { [] }> = unsafe { load_f16_ptr(ptrs, group_id) };
        let rows: i32 = unsafe {
            assume_div_by::<_, 16>(assume_bounds_lower::<_, 0>(tile_to_scalar(rows_tile)))
        };
        let cols: i32 = unsafe {
            assume_div_by::<_, 16>(assume_bounds_lower::<_, 0>(tile_to_scalar(cols_tile)))
        };
        let stride0: i32 = unsafe {
            assume_div_by::<_, 8>(assume_bounds_lower::<_, 0>(tile_to_scalar(stride0_tile)))
        };
        let shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[rows, cols],
        };
        let strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[stride0] };
        let tensor: Tensor<f16, { [-1, -1] }> =
            unsafe { make_tensor_view(ptr, shape, strides, new_token_unordered()) };
        (tensor, rows, cols)
    }

    // TileGym-style persistent group GEMM over vectors of tensor pointers.
    // Tensor extents vary per group through compact metadata tables; BM/BN/BK
    // remain compile-time tile shapes, so the host buckets by tile shape.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, num_cta_in_cga=2, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn group_gemm_f16_nt_desc<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const NUM_SM: i32,
    >(
        a_ptrs: &Tensor<i64, { [-1] }>,
        b_ptrs: &Tensor<i64, { [-1] }>,
        c_ptrs: &Tensor<i64, { [-1] }>,
        a_metas: &Tensor<i32, { [-1, 8] }>,
        b_metas: &Tensor<i32, { [-1, 8] }>,
        c_metas: &Tensor<i32, { [-1, 8] }>,
        num_groups: i32,
    ) {
        let mut tile_idx: i32 = get_tile_block_id().0;
        let mut last_problem_end: i32 = 0;

        for group_id in 0i32..num_groups {
            let (a, m, k): (Tensor<f16, { [-1, -1] }>, i32, i32) =
                unsafe { load_f16_desc_2d(a_ptrs, a_metas, group_id) };
            let (b, _bk_rows, n): (Tensor<f16, { [-1, -1] }>, i32, i32) =
                unsafe { load_f16_desc_2d(b_ptrs, b_metas, group_id) };
            let (c, _cm, _cn): (Tensor<f16, { [-1, -1] }>, i32, i32) =
                unsafe { load_f16_desc_2d(c_ptrs, c_metas, group_id) };

            let num_m_tiles: i32 = ceil_div(m, BM);
            let num_n_tiles: i32 = ceil_div(n, BN);
            let num_k_tiles: i32 = ceil_div(k, BK);
            let num_tiles: i32 = num_m_tiles * num_n_tiles;

            let a_part: Partition<f16, { [BM, BK] }> = a.partition(const_shape![BM, BK]);
            let b_part: Partition<f16, { [BK, BN] }> = b.partition(const_shape![BK, BN]);
            let mut c_part: PartitionMut<f16, { [BM, BN] }> =
                unsafe { c.partition_full_mut(const_shape![BM, BN]) };

            while tile_idx >= last_problem_end && tile_idx < last_problem_end + num_tiles {
                let tile_idx_in_group: i32 = tile_idx - last_problem_end;
                let tile_m_idx: i32 = tile_idx_in_group / num_n_tiles;
                let tile_n_idx: i32 = tile_idx_in_group - tile_m_idx * num_n_tiles;

                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                for kk in 0i32..num_k_tiles {
                    let ta: Tile<f16, { [BM, BK] }> = a_part.load([tile_m_idx, kk]);
                    let tb: Tile<f16, { [BK, BN] }> = b_part.load([kk, tile_n_idx]);
                    acc = mma(ta, tb, acc);
                }

                let out: Tile<f16, { [BM, BN] }> = convert_tile(acc);
                unsafe {
                    c_part.store(out, [tile_m_idx, tile_n_idx]);
                }

                tile_idx = tile_idx + NUM_SM;
            }

            last_problem_end = last_problem_end + num_tiles;
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn add_vec_f16<const S: [i32; 1]>(
        out: &mut Tensor<f16, S>,
        lhs: &Tensor<f16, { [-1] }>,
        rhs: &Tensor<f16, { [-1] }>,
    ) {
        let lhs_tile = load_tile_like(lhs, out);
        let rhs_tile = load_tile_like(rhs, out);
        out.store(lhs_tile + rhs_tile);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn add_2d_f16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        lhs: &Tensor<f16, { [-1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let lhs_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            lhs.partition(const_shape![1, BLOCK_SIZE]);
        let rhs_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            rhs.partition(const_shape![1, BLOCK_SIZE]);
        let lhs_tile: Tile<f16, { [1, BLOCK_SIZE] }> = lhs_part.load([row, col]);
        let rhs_tile: Tile<f16, { [1, BLOCK_SIZE] }> = rhs_part.load([row, col]);
        out.store(lhs_tile + rhs_tile);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn silu_mul_vec_f16<const S: [i32; 1]>(
        out: &mut Tensor<f16, S>,
        gate: &Tensor<f16, { [-1] }>,
        up: &Tensor<f16, { [-1] }>,
    ) {
        let gate_f16 = load_tile_like(gate, out);
        let up_f16 = load_tile_like(up, out);
        let gate_f32: Tile<f32, S> = convert_tile(gate_f16);
        let up_f32: Tile<f32, S> = convert_tile(up_f16);
        let one: Tile<f32, S> = constant(1.0f32, out.shape());
        let zero: Tile<f32, S> = constant(0.0f32, out.shape());
        let neg_gate: Tile<f32, S> = zero - gate_f32;
        let exp_neg_gate: Tile<f32, S> = exp(neg_gate);
        let denom: Tile<f32, S> = one + exp_neg_gate;
        let sigmoid: Tile<f32, S> = true_div(one, denom);
        let y: Tile<f32, S> = sigmoid * gate_f32 * up_f32;
        let y_f16: Tile<f16, S> = convert_tile(y);
        out.store(y_f16);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn silu_mul_2d_f16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        gate: &Tensor<f16, { [-1, -1] }>,
        up: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let gate_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            gate.partition(const_shape![1, BLOCK_SIZE]);
        let up_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            up.partition(const_shape![1, BLOCK_SIZE]);
        let gate_f16: Tile<f16, { [1, BLOCK_SIZE] }> = gate_part.load([row, col]);
        let up_f16: Tile<f16, { [1, BLOCK_SIZE] }> = up_part.load([row, col]);
        let gate_f32: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(gate_f16);
        let up_f32: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(up_f16);
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, const_shape![1, BLOCK_SIZE]);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, const_shape![1, BLOCK_SIZE]);
        let neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = zero - gate_f32;
        let exp_neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = exp(neg_gate);
        let denom: Tile<f32, { [1, BLOCK_SIZE] }> = one + exp_neg_gate;
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, denom);
        let y: Tile<f32, { [1, BLOCK_SIZE] }> = sigmoid * gate_f32 * up_f32;
        let y_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(y);
        out.store(y_f16);
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn rms_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
            unsafe { out_part.store(tout_f16, [0i32, j]) };
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn argmax_blocks_f16<const BLOCK_SIZE: i32>(
        logits: &Tensor<f16, { [-1] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<u32, { [1] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let logits_part = logits.partition(const_shape![BLOCK_SIZE]);
        let logits_f16: Tile<f16, { [BLOCK_SIZE] }> = logits_part.load([block]);
        let logits: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(logits_f16);

        let base: i32 = block * BLOCK_SIZE;
        let base_tile: Tile<i32, { [BLOCK_SIZE] }> = base.broadcast(const_shape![BLOCK_SIZE]);
        let offs: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let indices: Tile<i32, { [BLOCK_SIZE] }> = base_tile + offs;

        let len_tile: Tile<i32, { [BLOCK_SIZE] }> = len.broadcast(const_shape![BLOCK_SIZE]);
        let valid: Tile<bool, { [BLOCK_SIZE] }> = lt_tile(indices, len_tile);

        let mask_mag: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let zero: Tile<f32, { [BLOCK_SIZE] }> = constant(0.0f32, const_shape![BLOCK_SIZE]);
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> = zero - mask_mag;
        let masked_logits: Tile<f32, { [BLOCK_SIZE] }> = select(valid, logits, neg_inf);

        let max_tile: Tile<f32, { [1] }> = reduce_max(masked_logits, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_tile.reshape(const_shape![]));

        let max_bcast: Tile<f32, { [BLOCK_SIZE] }> = max_scalar.broadcast(const_shape![BLOCK_SIZE]);
        let is_max: Tile<bool, { [BLOCK_SIZE] }> = eq_tile(masked_logits, max_bcast);
        let invalid_idx: i32 = len + 1i32;
        let invalid_idx: Tile<i32, { [BLOCK_SIZE] }> =
            invalid_idx.broadcast(const_shape![BLOCK_SIZE]);
        let candidate_idx: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, indices, invalid_idx);
        let idx_tile: Tile<i32, { [1] }> = reduce_min(candidate_idx, 0i32);
        let idx_scalar: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let out_max_scalar: Tile<f32, { [] }> = scalar_to_tile(max_scalar);
        let out_max_tile: Tile<f32, { [1] }> = out_max_scalar.reshape(const_shape![1]);
        let out_idx_scalar: Tile<i32, { [] }> = scalar_to_tile(idx_scalar);
        let out_idx_i32: Tile<i32, { [1] }> = out_idx_scalar.reshape(const_shape![1]);
        let out_idx_tile: Tile<u32, { [1] }> = bitcast(out_idx_i32);
        block_max.store(out_max_tile);
        block_idx.store(out_idx_tile);
    }

    // Prototype greedy path for tied LM-head decode: compute a block of vocab
    // logits and reduce to one local argmax, without materializing logits.
    //
    // weights: [vocab, K], hidden: [1, K]
    // Grid: ceil(vocab / 64). A second argmax_reduce_blocks_to_u32 launch
    // reduces the per-block maxima to token_ids[0].
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn lm_head_argmax_blocks_f16<const K: i32>(
        weights: &Tensor<f16, { [-1, K] }>,
        hidden: &Tensor<f16, { [1, K] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<u32, { [1] }>,
        vocab_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let rows_shape: Shape<{ [64] }> = const_shape![64];
        let weight_shape: Shape<{ [64, 32] }> = const_shape![64, 32];
        let hidden_shape: Shape<{ [1, 32] }> = const_shape![1, 32];
        let weight_part: Partition<f16, { [64, 32] }> = weights.partition(weight_shape);
        let hidden_part: Partition<f16, { [1, 32] }> = hidden.partition(hidden_shape);

        let mut acc: Tile<f32, { [64] }> = constant(0.0f32, rows_shape);
        let num_k_tiles: i32 = K / 32i32;
        for k_block in 0i32..num_k_tiles {
            let w_f16: Tile<f16, { [64, 32] }> = weight_part.load([block, k_block]);
            let h_f16: Tile<f16, { [1, 32] }> = hidden_part.load([0i32, k_block]);
            let w: Tile<f32, { [64, 32] }> = convert_tile(w_f16);
            let h: Tile<f32, { [1, 32] }> = convert_tile(h_f16);
            let h_bc: Tile<f32, { [64, 32] }> = h.broadcast(weight_shape);
            let prod: Tile<f32, { [64, 32] }> = w * h_bc;
            let partial: Tile<f32, { [64] }> = reduce_sum(prod, 1i32);
            acc = acc + partial;
        }

        let base: i32 = block * 64i32;
        let base_tile: Tile<i32, { [64] }> = base.broadcast(rows_shape);
        let offs: Tile<i32, { [64] }> = iota(rows_shape);
        let indices: Tile<i32, { [64] }> = base_tile + offs;

        let vocab_tile: Tile<i32, { [64] }> = vocab_size.broadcast(rows_shape);
        let valid: Tile<bool, { [64] }> = lt_tile(indices, vocab_tile);
        let mask_mag: Tile<f32, { [64] }> = constant(1.0e30f32, rows_shape);
        let zero: Tile<f32, { [64] }> = constant(0.0f32, rows_shape);
        let neg_inf: Tile<f32, { [64] }> = zero - mask_mag;
        let masked_logits: Tile<f32, { [64] }> = select(valid, acc, neg_inf);

        let max_tile: Tile<f32, { [1] }> = reduce_max(masked_logits, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_tile.reshape(const_shape![]));
        let max_bcast: Tile<f32, { [64] }> = max_scalar.broadcast(rows_shape);
        let is_max: Tile<bool, { [64] }> = eq_tile(masked_logits, max_bcast);

        let invalid_idx: i32 = vocab_size + 1i32;
        let invalid_idx: Tile<i32, { [64] }> = invalid_idx.broadcast(rows_shape);
        let candidate_idx: Tile<i32, { [64] }> = select(is_max, indices, invalid_idx);
        let idx_tile: Tile<i32, { [1] }> = reduce_min(candidate_idx, 0i32);
        let idx_scalar: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let out_max_scalar: Tile<f32, { [] }> = scalar_to_tile(max_scalar);
        let out_max_tile: Tile<f32, { [1] }> = out_max_scalar.reshape(const_shape![1]);
        let out_idx_scalar: Tile<i32, { [] }> = scalar_to_tile(idx_scalar);
        let out_idx_i32: Tile<i32, { [1] }> = out_idx_scalar.reshape(const_shape![1]);
        let out_idx_tile: Tile<u32, { [1] }> = bitcast(out_idx_i32);
        block_max.store(out_max_tile);
        block_idx.store(out_idx_tile);
    }

    // Second pass of the two-stage argmax: reduce per-block
    // (block_max, block_idx) arrays to a single winning token id and store it
    // into `out[0]`. Single-CTA kernel — BLOCK_SIZE must be ≥ num_blocks.
    // For Qwen3 vocab=151936 with argmax_blocks BLOCK_SIZE=128,
    // num_blocks=1188, so BLOCK_SIZE=2048 fits.
    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn argmax_reduce_blocks_to_u32<const BLOCK_SIZE: i32>(
        block_max: &Tensor<f32, { [-1] }>,
        block_idx: &Tensor<u32, { [-1] }>,
        out: &mut Tensor<u32, { [1] }>,
        num_blocks: i32,
    ) {
        let bm_part = block_max.partition(const_shape![BLOCK_SIZE]);
        let bi_part = block_idx.partition(const_shape![BLOCK_SIZE]);

        let bm_tile: Tile<f32, { [BLOCK_SIZE] }> = bm_part.load([0i32]);
        let bi_tile_u32: Tile<u32, { [BLOCK_SIZE] }> = bi_part.load([0i32]);
        let bi_tile_i32: Tile<i32, { [BLOCK_SIZE] }> = bitcast(bi_tile_u32);

        let offs: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let n_tile: Tile<i32, { [BLOCK_SIZE] }> = num_blocks.broadcast(const_shape![BLOCK_SIZE]);
        let valid: Tile<bool, { [BLOCK_SIZE] }> = lt_tile(offs, n_tile);

        let mask_mag: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let zero_f: Tile<f32, { [BLOCK_SIZE] }> = constant(0.0f32, const_shape![BLOCK_SIZE]);
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> = zero_f - mask_mag;
        let masked_max: Tile<f32, { [BLOCK_SIZE] }> = select(valid, bm_tile, neg_inf);

        let max_t: Tile<f32, { [1] }> = reduce_max(masked_max, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_t.reshape(const_shape![]));
        let max_bcast: Tile<f32, { [BLOCK_SIZE] }> = max_scalar.broadcast(const_shape![BLOCK_SIZE]);
        let is_max: Tile<bool, { [BLOCK_SIZE] }> = eq_tile(masked_max, max_bcast);

        let big_idx: Tile<i32, { [BLOCK_SIZE] }> =
            constant(2147483647i32, const_shape![BLOCK_SIZE]);
        let candidates: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, bi_tile_i32, big_idx);
        let winner_t: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let winner: i32 = tile_to_scalar(winner_t.reshape(const_shape![]));

        let out_s: Tile<i32, { [] }> = scalar_to_tile(winner);
        let out_i32: Tile<i32, { [1] }> = out_s.reshape(const_shape![1]);
        let out_u32: Tile<u32, { [1] }> = bitcast(out_i32);
        out.store(out_u32);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn gather_row_f16<const BLOCK_SIZE: i32>(
        src: &Tensor<f16, { [-1, -1] }>,
        out: &mut Tensor<f16, { [BLOCK_SIZE] }>,
        row_idx: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;
        let src_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            src.partition(const_shape![1, BLOCK_SIZE]);
        let tile: Tile<f16, { [1, BLOCK_SIZE] }> = src_part.load([row_idx, block]);
        out.store(tile.reshape(const_shape![BLOCK_SIZE]));
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_f16<const D: i32, const HALF_D: i32>(
        x: &mut Tensor<f16, { [1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position: i32,
    ) {
        // Qwen3 text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, HALF_D] }> = x.partition(const_shape![1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, HALF_D] }> = x_part.load([0i32, 0i32]);
        let x_hi_f16: Tile<f16, { [1, HALF_D] }> = x_part.load([0i32, 1i32]);
        let x_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(x_hi_f16);

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos: f32 = convert_scalar(position);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, HALF_D] }> = theta.reshape(const_shape![1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_hi);

        let mut x_out = unsafe { x.partition_mut(const_shape![1, HALF_D]) };
        unsafe {
            x_out.store(y_lo_f16, [0i32, 0i32]);
            x_out.store(y_hi_f16, [0i32, 1i32]);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_seq_f16<const D: i32, const HALF_D: i32>(
        x: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
        position_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        // Qwen3 text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 1i32]);
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = position_start + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn rope_seq_dynpos_f16<const D: i32, const HALF_D: i32>(
        x: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position_start: &Tensor<u32, { [1] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        // Qwen3 text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 1i32]);
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        let pos_part = position_start.partition(const_shape![1]);
        let base_pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let base_pos_t: Tile<i32, { [1] }> = bitcast(base_pos_t_u32);
        let base_pos: i32 = tile_to_scalar(base_pos_t.reshape(const_shape![]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = base_pos + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn embedding_f16<const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(const_shape![1, BLOCK_SIZE]) };
        for j in 0i32..(D / BLOCK_SIZE) {
            let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, j]);
            unsafe { out_part.store(emb, [0i32, j]) };
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn embedding_batch_f16<const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let d_block = pid.1;

        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, d_block]);
        out.store(emb);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    fn kv_cache_update_f16<const D: i32, const BLOCK_SIZE: i32, const MAX_SEQ: i32>(
        new_k: &Tensor<f16, { [-1, D] }>,
        new_v: &Tensor<f16, { [-1, D] }>,
        k_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        position: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let d_block = pid.2;

        let new_k_part = new_k.partition(const_shape![1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, BLOCK_SIZE]);
        let mut k_cache_part = unsafe { k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };
        let mut v_cache_part = unsafe { v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };

        let k_tile = new_k_part
            .load([head, d_block])
            .reshape(const_shape![1, 1, BLOCK_SIZE]);
        let v_tile = new_v_part
            .load([head, d_block])
            .reshape(const_shape![1, 1, BLOCK_SIZE]);
        unsafe {
            k_cache_part.store(k_tile, [0i32, position, 0i32]);
            v_cache_part.store(v_tile, [0i32, position, 0i32]);
        }
    }

    // GEMM-like copy loop, parallelized across the time dimension. The
    // call site partitions the cache as [1, BM_S, VEC_BLOCK] so the
    // grid is (num_kv_heads, max_seq_len/BM_S, 1). Each CTA handles a
    // BM_S-token chunk via its per-CTA tile view; the inner loop is
    // BM_S iterations (vs seq_len before). Grid now scales with
    // max_seq_len instead of being pinned to num_kv_heads=8.
    //
    // Each CTA owns absolute cache slots; translate those slots back into
    // query rows. This also supports unaligned, nonzero MTP suffix writes.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (num_cta_in_cga=2, max_divisibility=16,),
                       ))]
    unsafe fn kv_cache_update_seq_f16<const D: i32, const BLOCK_SIZE: i32, const BM_S: i32>(
        new_k: &Tensor<f16, { [-1, -1, D] }>,
        new_v: &Tensor<f16, { [-1, -1, D] }>,
        k_cache: &mut Tensor<f16, { [1, BM_S, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, BM_S, BLOCK_SIZE] }>,
        position_start: i32,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let s_tile_idx = pid.1;
        let d_block = pid.2;

        let new_k_part = new_k.partition(const_shape![1, 1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut k_cache_part = unsafe { k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };
        let mut v_cache_part = unsafe { v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };

        let s_start: i32 = s_tile_idx * BM_S;
        // Skip trailing CTAs that are entirely beyond seq_len. The
        // per-CTA tile view naturally covers absolute cache positions
        // [s_start, s_start + BM_S) so indexing is local [0, BM_S).
        if s_start < position_start + seq_len && s_start + BM_S > position_start {
            for s_local in 0i32..BM_S {
                let s_global: i32 = s_start + s_local;
                if s_global >= position_start && s_global < position_start + seq_len {
                    let input_row: i32 = s_global - position_start;
                    let k_tile = new_k_part
                        .load([input_row, head, d_block])
                        .reshape(const_shape![1, 1, BLOCK_SIZE]);
                    let v_tile = new_v_part
                        .load([input_row, head, d_block])
                        .reshape(const_shape![1, 1, BLOCK_SIZE]);
                    // Cache storage uses the local offset in this CTA.
                    unsafe {
                        k_cache_part.store(k_tile, [0i32, s_local, 0i32]);
                        v_cache_part.store(v_tile, [0i32, s_local, 0i32]);
                    }
                }
            }
        }
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn kv_cache_update_seq_dynpos_f16<
        const D: i32,
        const BLOCK_SIZE: i32,
        const MAX_SEQ: i32,
    >(
        new_k: &Tensor<f16, { [-1, -1, D] }>,
        new_v: &Tensor<f16, { [-1, -1, D] }>,
        k_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        position_start: &Tensor<u32, { [1] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let d_block = pid.2;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let pos_start: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let new_k_part = new_k.partition(const_shape![1, 1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut k_cache_part = unsafe { k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };
        let mut v_cache_part = unsafe { v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };

        for s in 0i32..seq_len {
            let k_tile = new_k_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let v_tile = new_v_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let cache_pos = pos_start + s;
            unsafe {
                k_cache_part.store(k_tile, [0i32, cache_pos, 0i32]);
                v_cache_part.store(v_tile, [0i32, cache_pos, 0i32]);
            }
        }
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn flash_attn_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        v: &Tensor<f16, { [-1, -1, D] }>,
        out: &mut Tensor<f16, { [1, BM, D] }>,
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_head_idx = pid.0;
        let q_m_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [1, BM, D] }> = q.partition(const_shape![1, BM, D]);
        let tq: Tile<f16, { [1, BM, D] }> = q_part.load([q_head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [1, BM, D] }> = convert_tile(acc.reshape(const_shape![1, BM, D]));
        out.store(acc);
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn flash_attn_causal_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        v: &Tensor<f16, { [-1, -1, D] }>,
        out: &mut Tensor<f16, { [1, BM, D] }>,
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_head_idx = pid.0;
        let q_m_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [1, BM, D] }> = q.partition(const_shape![1, BM, D]);
        let tq: Tile<f16, { [1, BM, D] }> = q_part.load([q_head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [1, BM, D] }> = convert_tile(acc.reshape(const_shape![1, BM, D]));
        out.store(acc);
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn flash_attn_causal_seq_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, d]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // [q_len, q_heads, d]
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }

    // Prefill / general causal attention, ported from
    // TileGym/src/tilegym/ops/cutile/attention.py fmha_kernel_impl.
    // Differences from flash_attn_causal_seq_f16:
    //  * exp2 in log2 space (GPU SFU path — typically faster than natural exp)
    //  * qk_scale fused with the m_ij subtract instead of an up-front multiply
    //  * occupancy=2 instead of 1 (better latency hiding on sm_120)
    //  * EVEN_K const generic: skips bounds mask when kv_len is exactly
    //    divisible by BN, letting the mask-free tiles use the fast path
    //
    // Grid: (q_m_tiles, num_q_heads)
    //   Per-CTA: processes BM queries × all relevant K/V tiles.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=2, max_divisibility=16,),
                         sm_120 = (occupancy=2, max_divisibility=16,),
                       ))]
    unsafe fn fmha_prefill_causal<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const CAUSAL: i32,
        const EVEN_K: i32,
        const LATENCY: i32, // pipeline depth for K/V load_from_view; tune per arch
    >(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, D]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, D]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, D]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // per-CTA [BM, 1, D]
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;

        // Scale to log2 base: exp2(x * s / log2) = exp(x * s). Scalar since
        // we fuse the multiply into the m_ij subtract inside the loop.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_log2: f32 = qk_scale / log2;
        let qk_scale_tile: Tile<f32, { [BM, BN] }> = qk_scale_log2.broadcast(const_shape![BM, BN]);
        let qk_scale_col: Tile<f32, { [BM, 1] }> = qk_scale_log2.broadcast(const_shape![BM, 1]);

        // Query position offsets for causal mask.
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m_1d: Tile<i32, { [BM] }> =
            offs_m_base.broadcast(const_shape![BM]) + iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m_1d
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        // KV-tile offsets (within a tile).
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let kv_len_tile: Tile<i32, { [BM, BN] }> = kv_len.broadcast(const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> =
            constant(0.0f32, const_shape![BM, BN]) - constant(1.0e30f32, const_shape![BM, BN]);

        // Accumulators (rank 2 to avoid the cutile const-generic unification
        // issues we hit in the split-K kernel).
        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        // Load Q tile (one CTA = BM queries for one head).
        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq_raw: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f16, { [BM, D] }> = tq_raw.reshape(const_shape![BM, D]);

        // Tile iteration bounds (match flash_attn_causal_seq_f16's loop, just
        // hoisted out of the inner body).
        let m_end: i32 = query_start + (q_m_idx + 1i32) * BM;
        let k_seqlen_tiles: i32 = kv_len / BN;
        let mut mask_start: i32 = k_seqlen_tiles;
        let mut tc: i32 = ceil_div(kv_len, BN);
        if CAUSAL == 1i32 {
            mask_start = (query_start + q_m_idx * BM) / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, kv_len), BN);
        }

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..tc {
            // QK^T via a [D, BN]-shape K transpose; accumulator stays f32.
            // Both K and V go through load_from_view with Some(LATENCY):
            // swept on sm_120 and the two-loads-pipelined config is flat
            // across LAT ∈ {0..4} at OCC=2 (~116 ms at pp=2048), while
            // K-plain introduces a cliff at LAT<3 (128 ms regression).
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            qk = mma(tq, k_trans, qk);

            // Causal + OOB mask only on tiles where it can be violated.
            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, const_shape![BM, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [BM, BN] }> = constant(true, const_shape![BM, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                qk = qk + select(mask, mask_true, mask_false);
            }

            // Online softmax in log2 space. Reduce BEFORE scaling; apply scale
            // inside the `qk * scale - m_ij` fused op (TileGym perf note).
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max_col: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let qk_max_scaled: Tile<f32, { [BM, 1] }> = qk_max_col * qk_scale_col;
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max_scaled);
            let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![BM, BN]);
            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);

            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![BM, D]);

            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        // Normalize and cast back to f16.
        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_safe: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        let acc_norm: Tile<f32, { [BM, D] }> = true_div(acc, l_safe.broadcast(const_shape![BM, D]));
        let out_tile: Tile<f16, { [BM, 1, D] }> =
            convert_tile(acc_norm.reshape(const_shape![BM, 1, D]));
        out.store(out_tile);
    }

    // Head-grouped GQA prefill attention. Packs GROUP q_heads into the
    // m-dimension so K/V are loaded ONCE per (kv_head, j) iteration and
    // amortized across GROUP queries — a 1/GROUP reduction in K/V
    // bandwidth versus fmha_prefill_causal (the structural gap at long pp
    // vs FlashInfer/vLLM).
    //
    // Q partition tile:   [BM, GROUP, D]   → grid = (q_len/BM, kv_heads)
    // K/V partition tile: [1, BN, D]       → loaded ONCE per j iteration
    // Out partition tile: [BM, GROUP, D]   → matches Q layout
    //
    // Internally the Q tile is flattened to [M_EFF, D] with M_EFF =
    // BM*GROUP. Row r in M_EFF corresponds to (m, g) = (r/GROUP, r%GROUP)
    // because the reshape is row-major over [BM, GROUP, D]. q_pos depends
    // only on m, so the causal mask for row r uses
    //     q_pos[r] = query_start + q_m_idx*BM + r/GROUP
    // built via iota(BM).broadcast([BM, GROUP]).reshape([M_EFF]).
    //
    // M_EFF is passed as a separate const generic because cutile's type
    // system doesn't support const arithmetic in shape literals.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=2, max_divisibility=16,),
                         sm_120 = (occupancy=2, max_divisibility=16,),
                       ))]
    unsafe fn fmha_prefill_gqa<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const GROUP: i32,
        const M_EFF: i32, // caller MUST pass BM * GROUP
        const CAUSAL: i32,
        const EVEN_K: i32,
        const LATENCY: i32, // pipeline depth for Q/K/V load_from_view (gemma_attention-style)
    >(
        q: &Tensor<f16, { [-1, -1, D] }>,          // [q_len, q_heads, D]
        k: &Tensor<f16, { [-1, -1, D] }>,          // [kv_heads, kv_len, D]
        v: &Tensor<f16, { [-1, -1, D] }>,          // [kv_heads, kv_len, D]
        out: &mut Tensor<f16, { [BM, GROUP, D] }>, // per-CTA [BM, GROUP, D]
        qk_scale: f32,
        // query_group_size = q_heads / kv_heads (Qwen3: 32/8=4). GROUP is
        // the packing factor and must divide query_group_size. When GROUP
        // == query_group_size (default), each grid-dim-1 index maps 1:1
        // to a kv_head. For smaller GROUP, multiple grid-1 indices share
        // the same kv_head: kv_head_idx = pid.1 * GROUP / query_group_size.
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let kv_head_idx = pid.1 * GROUP / query_group_size;

        // Scale to log2 base: exp2(x * s / log2) = exp(x * s). Fused into
        // the `qk * scale - m_ij` op below.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_log2: f32 = qk_scale / log2;
        let qk_scale_tile: Tile<f32, { [M_EFF, BN] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, BN]);
        let qk_scale_col: Tile<f32, { [M_EFF, 1] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, 1]);

        // Build offs_m so row r has value query_start + q_m_idx*BM + r/GROUP:
        // iota(BM) reshaped to [BM, 1] and broadcast to [BM, GROUP], then
        // reshaped to [M_EFF, 1] yields [0,…,0, 1,…,1, …, BM-1,…] in
        // row-major order — exactly r/GROUP.
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let iota_bm: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let iota_bm_col: Tile<i32, { [BM, 1] }> = iota_bm.reshape(const_shape![BM, 1]);
        let iota_bm_grp: Tile<i32, { [BM, GROUP] }> =
            iota_bm_col.broadcast(const_shape![BM, GROUP]);
        let base_bg: Tile<i32, { [BM, GROUP] }> = offs_m_base.broadcast(const_shape![BM, GROUP]);
        let offs_m_bg: Tile<i32, { [BM, GROUP] }> = base_bg + iota_bm_grp;
        let offs_m_col: Tile<i32, { [M_EFF, 1] }> = offs_m_bg.reshape(const_shape![M_EFF, 1]);
        let offs_m: Tile<i32, { [M_EFF, BN] }> = offs_m_col.broadcast(const_shape![M_EFF, BN]);

        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [M_EFF, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![M_EFF, BN]);
        let kv_len_tile: Tile<i32, { [M_EFF, BN] }> = kv_len.broadcast(const_shape![M_EFF, BN]);
        let mask_false: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN])
            - constant(1.0e30f32, const_shape![M_EFF, BN]);

        // Rank-2 accumulators (match fmha_prefill_causal / decode-split
        // convention to dodge cutile const-generic unification issues).
        let max_mag: Tile<f32, { [M_EFF, 1] }> = constant(1.0e30f32, const_shape![M_EFF, 1]);
        let mut m_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]) - max_mag;
        let mut l_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]);
        let mut acc: Tile<f32, { [M_EFF, D] }> = constant(0.0f32, const_shape![M_EFF, D]);

        // Load Q tile once: [BM, GROUP, D] → [M_EFF, D]. Pipelined via
        // load_from_view with Some(LATENCY) — mirrors gemma_attention's
        // reference pattern which hints Q, K, V uniformly.
        let q_part: Partition<f16, { [BM, GROUP, D] }> = q.partition(const_shape![BM, GROUP, D]);
        let tq_raw: Tile<f16, { [BM, GROUP, D] }> = load_view_tko(
            &q_part,
            [q_m_idx, kv_head_idx, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let tq: Tile<f16, { [M_EFF, D] }> = tq_raw.reshape(const_shape![M_EFF, D]);

        // Tile iteration bounds. All GROUP queries at a given m share the
        // same q_pos, so the max q_pos in this CTA is still
        // (query_start + (q_m_idx+1)*BM - 1) — same as the non-grouped
        // kernel; group index doesn't affect the KV upper bound.
        let m_end: i32 = query_start + (q_m_idx + 1i32) * BM;
        let k_seqlen_tiles: i32 = kv_len / BN;
        let mut mask_start: i32 = k_seqlen_tiles;
        let mut tc: i32 = ceil_div(kv_len, BN);
        if CAUSAL == 1i32 {
            mask_start = (query_start + q_m_idx * BM) / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, kv_len), BN);
        }

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..tc {
            // ONE K load per iteration, reused across all GROUP queries.
            // Pipelined via load_from_view with Some(LATENCY) (reference
            // gemma_attention hints all of Q/K/V uniformly).
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let mut qk: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN]);
            qk = mma(tq, k_trans, qk);

            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [M_EFF, BN] }> =
                    broadcast_scalar(j * BN, const_shape![M_EFF, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [M_EFF, BN] }> = constant(true, const_shape![M_EFF, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [M_EFF, BN] }> = lt_tile(offs_n, kv_len_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [M_EFF, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                let mask_true: Tile<f32, { [M_EFF, BN] }> =
                    constant(0.0f32, const_shape![M_EFF, BN]);
                qk = qk + select(mask, mask_true, mask_false);
            }

            // Online softmax in log2 space (rowwise over M_EFF).
            let qk_max: Tile<f32, { [M_EFF] }> = reduce_max(qk, 1i32);
            let qk_max_col: Tile<f32, { [M_EFF, 1] }> = qk_max.reshape(const_shape![M_EFF, 1]);
            let qk_max_scaled: Tile<f32, { [M_EFF, 1] }> = qk_max_col * qk_scale_col;
            let m_ij: Tile<f32, { [M_EFF, 1] }> = max_tile(m_i, qk_max_scaled);
            let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![M_EFF, BN]);
            let p: Tile<f32, { [M_EFF, BN] }> = exp2(qk, ftz::Disabled);

            let l_ij: Tile<f32, { [M_EFF] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [M_EFF, 1] }> = l_ij.reshape(const_shape![M_EFF, 1]);
            let alpha: Tile<f32, { [M_EFF, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![M_EFF, D]);

            // ONE V load per iteration, reused across all GROUP queries.
            // Pipelined via load_from_view with Some(LATENCY).
            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let p_f16: Tile<f16, { [M_EFF, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        // Normalize and reshape acc [M_EFF, D] → [BM, GROUP, D] for store.
        let eps: Tile<f32, { [M_EFF, 1] }> = constant(1.0e-8f32, const_shape![M_EFF, 1]);
        let l_safe: Tile<f32, { [M_EFF, 1] }> = max_tile(l_i, eps);
        let acc_norm: Tile<f32, { [M_EFF, D] }> =
            true_div(acc, l_safe.broadcast(const_shape![M_EFF, D]));
        let out_tile: Tile<f16, { [BM, GROUP, D] }> =
            convert_tile(acc_norm.reshape(const_shape![BM, GROUP, D]));
        out.store(out_tile);
    }

    // TileGym-style LPT/swizzled GQA prefill. This keeps the same grouped
    // high-level math as fmha_prefill_gqa, but uses raw Q/K/V/O pointers so
    // the physical CTA schedule can be changed independently of the logical
    // tensor partition order.
    //
    // SCHED:
    //   0: swizzled q-block-major, reverse q-block order (current LPT)
    //   1: plain q-block-major, reverse q-block order
    //   2: head-group-major, reverse q-block order
    //   3: swizzled q-block-major, forward q-block order
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=2, max_divisibility=16,),
                         sm_120 = (occupancy=2, max_divisibility=16,),
                       ))]
    unsafe fn fmha_prefill_gqa_lpt<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const GROUP: i32,
        const M_EFF: i32, // caller MUST pass BM * GROUP
        const CAUSAL: i32,
        const EVEN_K: i32,
        const LATENCY: i32,
        const SCHED: i32,
        const MASK_SPLIT: i32,
    >(
        q_ptr: *mut f16,   // [q_len, q_heads, D]
        k_ptr: *mut f16,   // [kv_heads, kv_len, D]
        v_ptr: *mut f16,   // [kv_heads, kv_len, D]
        out_ptr: *mut f16, // [q_len, q_heads, D]
        qk_scale: f32,
        query_group_size: i32,
        q_len: i32,
        kv_len: i32,
        query_start: i32,
        num_q_blocks: i32,
        num_head_groups: i32,
        swizzle: i32,
        num_hb_quotient: i32,
        num_hb_remainder: i32,
    ) {
        let q_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_ptr) };
        let k_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_ptr) };
        let v_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_ptr) };
        let out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(out_ptr) };
        let q_len: i32 = unsafe { assume_bounds_lower::<_, 0>(q_len) };
        let kv_len: i32 = unsafe { assume_bounds_lower::<_, 0>(kv_len) };
        let num_head_groups: i32 = unsafe { assume_bounds_lower::<_, 0>(num_head_groups) };

        let tok: Token = new_token_unordered();
        let q_heads: i32 = num_head_groups * GROUP;
        let kv_heads: i32 = q_heads / query_group_size;
        let q_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[q_len, q_heads],
        };
        let q_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[q_heads * D, D],
        };
        let q_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_ptr), q_shape, q_strides, tok) };
        let kv_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[kv_heads, kv_len],
        };
        let kv_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[kv_len * D, D],
        };
        let k_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_ptr), kv_shape, kv_strides, tok) };
        let v_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(v_ptr), kv_shape, kv_strides, tok) };
        let out_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[q_len, q_heads],
        };
        let out_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[q_heads * D, D],
        };
        let out_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(out_ptr), out_shape, out_strides, tok) };

        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_idx = pid.0;
        let total_tiles: i32 = num_q_blocks * num_head_groups;
        if tile_idx >= total_tiles {
            return;
        }

        let sched: (i32, i32, i32) = if SCHED == 1i32 {
            {
                // Plain q-block-major order: all head groups for a q block,
                // then the next shorter q block.
                let block: i32 = tile_idx / num_head_groups;
                let q_head_group_idx: i32 = tile_idx - block * num_head_groups;
                (block, q_head_group_idx, 1i32)
            }
        } else {
            if SCHED == 2i32 {
                {
                    // Head-group-major order: complete the LPT q-block walk
                    // for one head group before moving to the next.
                    let q_head_group_idx: i32 = tile_idx / num_q_blocks;
                    let block: i32 = tile_idx - q_head_group_idx * num_q_blocks;
                    (block, q_head_group_idx, 1i32)
                }
            } else {
                {
                    // Same swizzle mapping as TileGym's ragged prefill
                    // launcher, specialized to one batch and q_head_group.
                    let l2_major_blocks: i32 = swizzle * num_q_blocks;
                    let bidhb: i32 = tile_idx / l2_major_blocks;
                    let l2_mod: i32 = tile_idx - bidhb * l2_major_blocks;
                    let head_group_span: i32 = if bidhb < num_hb_quotient {
                        swizzle
                    } else {
                        num_hb_remainder
                    };
                    let block: i32 = l2_mod / head_group_span;
                    let bidhb_residual: i32 = l2_mod - block * head_group_span;
                    let q_head_group_idx: i32 = bidhb * swizzle + bidhb_residual;
                    let reverse: i32 = if SCHED == 3i32 { 0i32 } else { 1i32 };
                    (block, q_head_group_idx, reverse)
                }
            }
        };
        let block: i32 = sched.0;
        let q_head_group_idx: i32 = sched.1;
        if q_head_group_idx >= num_head_groups {
            return;
        }
        let q_m_idx: i32 = if sched.2 == 1i32 {
            num_q_blocks - 1i32 - block
        } else {
            block
        };
        let kv_head_idx: i32 = q_head_group_idx * GROUP / query_group_size;

        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_log2: f32 = qk_scale / log2;
        let qk_scale_tile: Tile<f32, { [M_EFF, BN] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, BN]);
        let qk_scale_col: Tile<f32, { [M_EFF, 1] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, 1]);

        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let iota_bm: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let iota_bm_col: Tile<i32, { [BM, 1] }> = iota_bm.reshape(const_shape![BM, 1]);
        let iota_bm_grp: Tile<i32, { [BM, GROUP] }> =
            iota_bm_col.broadcast(const_shape![BM, GROUP]);
        let base_bg: Tile<i32, { [BM, GROUP] }> = offs_m_base.broadcast(const_shape![BM, GROUP]);
        let offs_m_bg: Tile<i32, { [BM, GROUP] }> = base_bg + iota_bm_grp;
        let offs_m_col: Tile<i32, { [M_EFF, 1] }> = offs_m_bg.reshape(const_shape![M_EFF, 1]);
        let offs_m: Tile<i32, { [M_EFF, BN] }> = offs_m_col.broadcast(const_shape![M_EFF, BN]);

        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [M_EFF, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![M_EFF, BN]);
        let kv_len_tile: Tile<i32, { [M_EFF, BN] }> = kv_len.broadcast(const_shape![M_EFF, BN]);
        let mask_false: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN])
            - constant(1.0e30f32, const_shape![M_EFF, BN]);

        let max_mag: Tile<f32, { [M_EFF, 1] }> = constant(1.0e30f32, const_shape![M_EFF, 1]);
        let mut m_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]) - max_mag;
        let mut l_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]);
        let mut acc: Tile<f32, { [M_EFF, D] }> = constant(0.0f32, const_shape![M_EFF, D]);

        let q_part: Partition<f16, { [BM, GROUP, D] }> =
            q_tv.partition_permuted(const_shape![BM, GROUP, D], const_array![0, 1, 2]);
        let tq_raw: Tile<f16, { [BM, GROUP, D] }> = load_view_tko(
            &q_part,
            [q_m_idx, q_head_group_idx, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let tq: Tile<f16, { [M_EFF, D] }> = tq_raw.reshape(const_shape![M_EFF, D]);

        let m_end: i32 = query_start + (q_m_idx + 1i32) * BM;
        let k_seqlen_tiles: i32 = kv_len / BN;
        let mut mask_start: i32 = k_seqlen_tiles;
        let mut tc: i32 = ceil_div(kv_len, BN);
        if CAUSAL == 1i32 {
            mask_start = (query_start + q_m_idx * BM) / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, kv_len), BN);
        }

        let k_part: Partition<f16, { [1, BN, D] }> =
            k_tv.partition_permuted(const_shape![1, BN, D], const_array![0, 1, 2]);
        let v_part: Partition<f16, { [1, BN, D] }> =
            v_tv.partition_permuted(const_shape![1, BN, D], const_array![0, 1, 2]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        if MASK_SPLIT == 1i32 && CAUSAL == 1i32 {
            for j in 0i32..mask_start {
                let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                    &k_part,
                    [kv_head_idx, j, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(LATENCY),
                    tma::Enabled,
                );
                let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
                let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
                let mut qk: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN]);
                qk = mma(tq, k_trans, qk);

                let qk_max: Tile<f32, { [M_EFF] }> = reduce_max(qk, 1i32);
                let qk_max_col: Tile<f32, { [M_EFF, 1] }> = qk_max.reshape(const_shape![M_EFF, 1]);
                let qk_max_scaled: Tile<f32, { [M_EFF, 1] }> = qk_max_col * qk_scale_col;
                let m_ij: Tile<f32, { [M_EFF, 1] }> = max_tile(m_i, qk_max_scaled);
                let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![M_EFF, BN]);
                let p: Tile<f32, { [M_EFF, BN] }> = exp2(qk, ftz::Disabled);

                let l_ij: Tile<f32, { [M_EFF] }> = reduce_sum(p, 1i32);
                let l_ij: Tile<f32, { [M_EFF, 1] }> = l_ij.reshape(const_shape![M_EFF, 1]);
                let alpha: Tile<f32, { [M_EFF, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
                l_i = l_i * alpha + l_ij;
                acc = acc * alpha.broadcast(const_shape![M_EFF, D]);

                let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                    &v_part,
                    [kv_head_idx, j, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(LATENCY),
                    tma::Enabled,
                );
                let p_f16: Tile<f16, { [M_EFF, BN] }> = convert_tile(p);
                let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
                acc = mma(p_f16, v_tile, acc);
                m_i = m_ij;
            }
            for j in mask_start..tc {
                let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                    &k_part,
                    [kv_head_idx, j, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(LATENCY),
                    tma::Enabled,
                );
                let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
                let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
                let mut qk: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN]);
                qk = mma(tq, k_trans, qk);

                let offs_n: Tile<i32, { [M_EFF, BN] }> =
                    broadcast_scalar(j * BN, const_shape![M_EFF, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [M_EFF, BN] }> = constant(true, const_shape![M_EFF, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [M_EFF, BN] }> = lt_tile(offs_n, kv_len_tile);
                    mask = mask & lt_res;
                }
                let ge_res: Tile<bool, { [M_EFF, BN] }> = ge_tile(offs_m, offs_n);
                mask = mask & ge_res;
                let mask_true: Tile<f32, { [M_EFF, BN] }> =
                    constant(0.0f32, const_shape![M_EFF, BN]);
                qk = qk + select(mask, mask_true, mask_false);

                let qk_max: Tile<f32, { [M_EFF] }> = reduce_max(qk, 1i32);
                let qk_max_col: Tile<f32, { [M_EFF, 1] }> = qk_max.reshape(const_shape![M_EFF, 1]);
                let qk_max_scaled: Tile<f32, { [M_EFF, 1] }> = qk_max_col * qk_scale_col;
                let m_ij: Tile<f32, { [M_EFF, 1] }> = max_tile(m_i, qk_max_scaled);
                let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![M_EFF, BN]);
                let p: Tile<f32, { [M_EFF, BN] }> = exp2(qk, ftz::Disabled);

                let l_ij: Tile<f32, { [M_EFF] }> = reduce_sum(p, 1i32);
                let l_ij: Tile<f32, { [M_EFF, 1] }> = l_ij.reshape(const_shape![M_EFF, 1]);
                let alpha: Tile<f32, { [M_EFF, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
                l_i = l_i * alpha + l_ij;
                acc = acc * alpha.broadcast(const_shape![M_EFF, D]);

                let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                    &v_part,
                    [kv_head_idx, j, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(LATENCY),
                    tma::Enabled,
                );
                let p_f16: Tile<f16, { [M_EFF, BN] }> = convert_tile(p);
                let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
                acc = mma(p_f16, v_tile, acc);
                m_i = m_ij;
            }
        } else {
            for j in 0i32..tc {
                let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                    &k_part,
                    [kv_head_idx, j, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(LATENCY),
                    tma::Enabled,
                );
                let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
                let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
                let mut qk: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN]);
                qk = mma(tq, k_trans, qk);

                if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                    let offs_n: Tile<i32, { [M_EFF, BN] }> =
                        broadcast_scalar(j * BN, const_shape![M_EFF, BN]) + offs_n_tile;
                    let mut mask: Tile<bool, { [M_EFF, BN] }> =
                        constant(true, const_shape![M_EFF, BN]);
                    if EVEN_K == 0i32 {
                        let lt_res: Tile<bool, { [M_EFF, BN] }> = lt_tile(offs_n, kv_len_tile);
                        mask = mask & lt_res;
                    }
                    if CAUSAL == 1i32 {
                        let ge_res: Tile<bool, { [M_EFF, BN] }> = ge_tile(offs_m, offs_n);
                        mask = mask & ge_res;
                    }
                    let mask_true: Tile<f32, { [M_EFF, BN] }> =
                        constant(0.0f32, const_shape![M_EFF, BN]);
                    qk = qk + select(mask, mask_true, mask_false);
                }

                let qk_max: Tile<f32, { [M_EFF] }> = reduce_max(qk, 1i32);
                let qk_max_col: Tile<f32, { [M_EFF, 1] }> = qk_max.reshape(const_shape![M_EFF, 1]);
                let qk_max_scaled: Tile<f32, { [M_EFF, 1] }> = qk_max_col * qk_scale_col;
                let m_ij: Tile<f32, { [M_EFF, 1] }> = max_tile(m_i, qk_max_scaled);
                let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![M_EFF, BN]);
                let p: Tile<f32, { [M_EFF, BN] }> = exp2(qk, ftz::Disabled);

                let l_ij: Tile<f32, { [M_EFF] }> = reduce_sum(p, 1i32);
                let l_ij: Tile<f32, { [M_EFF, 1] }> = l_ij.reshape(const_shape![M_EFF, 1]);
                let alpha: Tile<f32, { [M_EFF, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
                l_i = l_i * alpha + l_ij;
                acc = acc * alpha.broadcast(const_shape![M_EFF, D]);

                let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                    &v_part,
                    [kv_head_idx, j, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(LATENCY),
                    tma::Enabled,
                );
                let p_f16: Tile<f16, { [M_EFF, BN] }> = convert_tile(p);
                let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
                acc = mma(p_f16, v_tile, acc);
                m_i = m_ij;
            }
        }

        let eps: Tile<f32, { [M_EFF, 1] }> = constant(1.0e-8f32, const_shape![M_EFF, 1]);
        let l_safe: Tile<f32, { [M_EFF, 1] }> = max_tile(l_i, eps);
        let acc_norm: Tile<f32, { [M_EFF, D] }> =
            true_div(acc, l_safe.broadcast(const_shape![M_EFF, D]));
        let out_tile: Tile<f16, { [BM, GROUP, D] }> =
            convert_tile(acc_norm.reshape(const_shape![BM, GROUP, D]));

        let mut out_part: PartitionMut<f16, { [BM, GROUP, D] }> =
            unsafe { out_tv.partition_full_mut(const_shape![BM, GROUP, D]) };
        unsafe {
            out_part.store(out_tile, [q_m_idx, q_head_group_idx, 0i32]);
        }
    }

    // Split-K prefill variant for the raw-pointer GQA LPT path. This writes
    // normalized per-split partial outputs plus per-row LSE into scratch.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=2, max_divisibility=16,),
                         sm_120 = (occupancy=2, max_divisibility=16,),
                       ))]
    unsafe fn fmha_prefill_gqa_lpt_split<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const GROUP: i32,
        const M_EFF: i32,
        const EVEN_K: i32,
        const NUM_KV_SPLITS: i32,
        const NS_M: i32,
        const LATENCY: i32,
        const SCHED: i32,
    >(
        q_ptr: *mut f16,
        k_ptr: *mut f16,
        v_ptr: *mut f16,
        att_partial_ptr: *mut f16, // [num_tiles, NUM_KV_SPLITS * M_EFF, D]
        lse_partial_ptr: *mut f32, // [num_tiles, NUM_KV_SPLITS * M_EFF]
        qk_scale: f32,
        query_group_size: i32,
        q_len: i32,
        kv_len: i32,
        query_start: i32,
        num_q_blocks: i32,
        num_head_groups: i32,
        swizzle: i32,
        num_hb_quotient: i32,
        num_hb_remainder: i32,
    ) {
        let q_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_ptr) };
        let k_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_ptr) };
        let v_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_ptr) };
        let att_partial_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(att_partial_ptr) };
        let q_len: i32 = unsafe { assume_bounds_lower::<_, 0>(q_len) };
        let kv_len: i32 = unsafe { assume_bounds_lower::<_, 0>(kv_len) };
        let num_head_groups: i32 = unsafe { assume_bounds_lower::<_, 0>(num_head_groups) };

        let tok: Token = new_token_unordered();
        let q_heads: i32 = num_head_groups * GROUP;
        let kv_heads: i32 = q_heads / query_group_size;
        let total_tiles: i32 = num_q_blocks * num_head_groups;

        let q_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[q_len, q_heads],
        };
        let q_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[q_heads * D, D],
        };
        let q_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_ptr), q_shape, q_strides, tok) };
        let kv_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[kv_heads, kv_len],
        };
        let kv_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[kv_len * D, D],
        };
        let k_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_ptr), kv_shape, kv_strides, tok) };
        let v_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(v_ptr), kv_shape, kv_strides, tok) };
        let att_shape: Shape<{ [-1, NS_M, D] }> = Shape::<{ [-1, NS_M, D] }> {
            dims: &[total_tiles],
        };
        let att_strides: Array<{ [-1, D, 1] }> = Array::<{ [-1, D, 1] }> { dims: &[NS_M * D] };
        let att_tv: Tensor<f16, { [-1, NS_M, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(att_partial_ptr),
                att_shape,
                att_strides,
                tok,
            )
        };
        let lse_shape: Shape<{ [-1, NS_M] }> = Shape::<{ [-1, NS_M] }> {
            dims: &[total_tiles],
        };
        let lse_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[NS_M] };
        let lse_tv: Tensor<f32, { [-1, NS_M] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(lse_partial_ptr),
                lse_shape,
                lse_strides,
                tok,
            )
        };

        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_idx = pid.0;
        let split_id = pid.1;
        if tile_idx >= total_tiles {
            return;
        }

        let sched: (i32, i32, i32) = if SCHED == 1i32 {
            {
                let block: i32 = tile_idx / num_head_groups;
                let q_head_group_idx: i32 = tile_idx - block * num_head_groups;
                (block, q_head_group_idx, 1i32)
            }
        } else {
            if SCHED == 2i32 {
                {
                    let q_head_group_idx: i32 = tile_idx / num_q_blocks;
                    let block: i32 = tile_idx - q_head_group_idx * num_q_blocks;
                    (block, q_head_group_idx, 1i32)
                }
            } else {
                {
                    let l2_major_blocks: i32 = swizzle * num_q_blocks;
                    let bidhb: i32 = tile_idx / l2_major_blocks;
                    let l2_mod: i32 = tile_idx - bidhb * l2_major_blocks;
                    let head_group_span: i32 = if bidhb < num_hb_quotient {
                        swizzle
                    } else {
                        num_hb_remainder
                    };
                    let block: i32 = l2_mod / head_group_span;
                    let bidhb_residual: i32 = l2_mod - block * head_group_span;
                    let q_head_group_idx: i32 = bidhb * swizzle + bidhb_residual;
                    let reverse: i32 = if SCHED == 3i32 { 0i32 } else { 1i32 };
                    (block, q_head_group_idx, reverse)
                }
            }
        };
        let block: i32 = sched.0;
        let q_head_group_idx: i32 = sched.1;
        if q_head_group_idx >= num_head_groups {
            return;
        }
        let q_m_idx: i32 = if sched.2 == 1i32 {
            num_q_blocks - 1i32 - block
        } else {
            block
        };
        let logical_tile_idx: i32 = q_m_idx * num_head_groups + q_head_group_idx;
        let kv_head_idx: i32 = q_head_group_idx * GROUP / query_group_size;

        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2_v: f32 = tile_to_scalar(log(two));
        let qk_scale_log2: f32 = qk_scale / log2_v;
        let qk_scale_tile: Tile<f32, { [M_EFF, BN] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, BN]);
        let qk_scale_col: Tile<f32, { [M_EFF, 1] }> =
            qk_scale_log2.broadcast(const_shape![M_EFF, 1]);

        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let iota_bm: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let iota_bm_col: Tile<i32, { [BM, 1] }> = iota_bm.reshape(const_shape![BM, 1]);
        let iota_bm_grp: Tile<i32, { [BM, GROUP] }> =
            iota_bm_col.broadcast(const_shape![BM, GROUP]);
        let base_bg: Tile<i32, { [BM, GROUP] }> = offs_m_base.broadcast(const_shape![BM, GROUP]);
        let offs_m_bg: Tile<i32, { [BM, GROUP] }> = base_bg + iota_bm_grp;
        let offs_m_col: Tile<i32, { [M_EFF, 1] }> = offs_m_bg.reshape(const_shape![M_EFF, 1]);
        let offs_m: Tile<i32, { [M_EFF, BN] }> = offs_m_col.broadcast(const_shape![M_EFF, BN]);
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [M_EFF, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![M_EFF, BN]);
        let kv_len_tile: Tile<i32, { [M_EFF, BN] }> = kv_len.broadcast(const_shape![M_EFF, BN]);
        let mask_false: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN])
            - constant(1.0e30f32, const_shape![M_EFF, BN]);

        let q_part: Partition<f16, { [BM, GROUP, D] }> =
            q_tv.partition_permuted(const_shape![BM, GROUP, D], const_array![0, 1, 2]);
        let tq_raw: Tile<f16, { [BM, GROUP, D] }> = load_view_tko(
            &q_part,
            [q_m_idx, q_head_group_idx, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let tq: Tile<f16, { [M_EFF, D] }> = tq_raw.reshape(const_shape![M_EFF, D]);

        let m_end: i32 = query_start + (q_m_idx + 1i32) * BM;
        let k_seqlen_tiles: i32 = kv_len / BN;
        let mut mask_start: i32 = (query_start + q_m_idx * BM) / BN;
        mask_start = min(mask_start, k_seqlen_tiles);
        let tc: i32 = ceil_div(min(m_end, kv_len), BN);
        let tiles_per_split: i32 = ceil_div(tc, NUM_KV_SPLITS);
        let start_tile: i32 = split_id * tiles_per_split;
        let mut end_tile: i32 = start_tile + tiles_per_split;
        end_tile = min(end_tile, tc);

        let k_part: Partition<f16, { [1, BN, D] }> =
            k_tv.partition_permuted(const_shape![1, BN, D], const_array![0, 1, 2]);
        let v_part: Partition<f16, { [1, BN, D] }> =
            v_tv.partition_permuted(const_shape![1, BN, D], const_array![0, 1, 2]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        let max_mag: Tile<f32, { [M_EFF, 1] }> = constant(1.0e30f32, const_shape![M_EFF, 1]);
        let mut m_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]) - max_mag;
        let mut l_i: Tile<f32, { [M_EFF, 1] }> = constant(0.0f32, const_shape![M_EFF, 1]);
        let mut acc: Tile<f32, { [M_EFF, D] }> = constant(0.0f32, const_shape![M_EFF, D]);

        for j in start_tile..end_tile {
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let mut qk: Tile<f32, { [M_EFF, BN] }> = constant(0.0f32, const_shape![M_EFF, BN]);
            qk = mma(tq, k_trans, qk);

            if j >= mask_start {
                let offs_n: Tile<i32, { [M_EFF, BN] }> =
                    broadcast_scalar(j * BN, const_shape![M_EFF, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [M_EFF, BN] }> = constant(true, const_shape![M_EFF, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [M_EFF, BN] }> = lt_tile(offs_n, kv_len_tile);
                    mask = mask & lt_res;
                }
                let ge_res: Tile<bool, { [M_EFF, BN] }> = ge_tile(offs_m, offs_n);
                mask = mask & ge_res;
                let mask_true: Tile<f32, { [M_EFF, BN] }> =
                    constant(0.0f32, const_shape![M_EFF, BN]);
                qk = qk + select(mask, mask_true, mask_false);
            }

            let qk_max: Tile<f32, { [M_EFF] }> = reduce_max(qk, 1i32);
            let qk_max_col: Tile<f32, { [M_EFF, 1] }> = qk_max.reshape(const_shape![M_EFF, 1]);
            let qk_max_scaled: Tile<f32, { [M_EFF, 1] }> = qk_max_col * qk_scale_col;
            let m_ij: Tile<f32, { [M_EFF, 1] }> = max_tile(m_i, qk_max_scaled);
            let qk = qk * qk_scale_tile - m_ij.broadcast(const_shape![M_EFF, BN]);
            let p: Tile<f32, { [M_EFF, BN] }> = exp2(qk, ftz::Disabled);

            let l_ij: Tile<f32, { [M_EFF] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [M_EFF, 1] }> = l_ij.reshape(const_shape![M_EFF, 1]);
            let alpha: Tile<f32, { [M_EFF, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![M_EFF, D]);

            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_idx, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let p_f16: Tile<f16, { [M_EFF, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [M_EFF, 1] }> = constant(1.0e-8f32, const_shape![M_EFF, 1]);
        let l_safe: Tile<f32, { [M_EFF, 1] }> = max_tile(l_i, eps);
        let acc_norm: Tile<f32, { [M_EFF, D] }> =
            true_div(acc, l_safe.broadcast(const_shape![M_EFF, D]));
        let att_tile: Tile<f16, { [1, M_EFF, D] }> =
            convert_tile(acc_norm.reshape(const_shape![1, M_EFF, D]));
        let mut att_part: PartitionMut<f16, { [1, M_EFF, D] }> =
            unsafe { att_tv.partition_full_mut(const_shape![1, M_EFF, D]) };
        unsafe {
            att_part.store(att_tile, [logical_tile_idx, split_id, 0i32]);
        }

        let lse_col: Tile<f32, { [M_EFF, 1] }> = m_i + log2(l_safe);
        let lse_tile: Tile<f32, { [1, M_EFF] }> = lse_col.reshape(const_shape![1, M_EFF]);
        let mut lse_part: PartitionMut<f32, { [1, M_EFF] }> =
            unsafe { lse_tv.partition_full_mut(const_shape![1, M_EFF]) };
        unsafe {
            lse_part.store(lse_tile, [logical_tile_idx, split_id]);
        }
    }

    // Merge prefill split-K partials into the final [q_len, q_heads, D] output.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=4, max_divisibility=16,),
                         sm_120 = (occupancy=4, max_divisibility=16,),
                       ))]
    unsafe fn prefill_splitk_reduce_merge<
        const BM: i32,
        const GROUP: i32,
        const D: i32,
        const M_EFF: i32,
        const CHUNK_D: i32,
        const NUM_KV_SPLITS: i32,
        const NS_M: i32,
        const SCHED: i32,
        const LATENCY: i32,
    >(
        att_partial_ptr: *mut f16,
        lse_partial_ptr: *mut f32,
        out_ptr: *mut f16,
        q_len: i32,
        num_q_blocks: i32,
        num_head_groups: i32,
        swizzle: i32,
        num_hb_quotient: i32,
        num_hb_remainder: i32,
    ) {
        let att_partial_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(att_partial_ptr) };
        let out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(out_ptr) };
        let q_len: i32 = unsafe { assume_bounds_lower::<_, 0>(q_len) };
        let num_head_groups: i32 = unsafe { assume_bounds_lower::<_, 0>(num_head_groups) };

        let tok: Token = new_token_unordered();
        let q_heads: i32 = num_head_groups * GROUP;
        let total_tiles: i32 = num_q_blocks * num_head_groups;
        let att_shape: Shape<{ [-1, NS_M, D] }> = Shape::<{ [-1, NS_M, D] }> {
            dims: &[total_tiles],
        };
        let att_strides: Array<{ [-1, D, 1] }> = Array::<{ [-1, D, 1] }> { dims: &[NS_M * D] };
        let att_tv: Tensor<f16, { [-1, NS_M, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(att_partial_ptr),
                att_shape,
                att_strides,
                tok,
            )
        };
        let lse_shape: Shape<{ [-1, NS_M] }> = Shape::<{ [-1, NS_M] }> {
            dims: &[total_tiles],
        };
        let lse_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[NS_M] };
        let lse_tv: Tensor<f32, { [-1, NS_M] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(lse_partial_ptr),
                lse_shape,
                lse_strides,
                tok,
            )
        };
        let out_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[q_len, q_heads],
        };
        let out_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[q_heads * D, D],
        };
        let out_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(out_ptr), out_shape, out_strides, tok) };

        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_idx = pid.0;
        let d_chunk_id = pid.1;
        if tile_idx >= total_tiles {
            return;
        }

        let sched: (i32, i32, i32) = if SCHED == 1i32 {
            {
                let block: i32 = tile_idx / num_head_groups;
                let q_head_group_idx: i32 = tile_idx - block * num_head_groups;
                (block, q_head_group_idx, 1i32)
            }
        } else {
            if SCHED == 2i32 {
                {
                    let q_head_group_idx: i32 = tile_idx / num_q_blocks;
                    let block: i32 = tile_idx - q_head_group_idx * num_q_blocks;
                    (block, q_head_group_idx, 1i32)
                }
            } else {
                {
                    let l2_major_blocks: i32 = swizzle * num_q_blocks;
                    let bidhb: i32 = tile_idx / l2_major_blocks;
                    let l2_mod: i32 = tile_idx - bidhb * l2_major_blocks;
                    let head_group_span: i32 = if bidhb < num_hb_quotient {
                        swizzle
                    } else {
                        num_hb_remainder
                    };
                    let block: i32 = l2_mod / head_group_span;
                    let bidhb_residual: i32 = l2_mod - block * head_group_span;
                    let q_head_group_idx: i32 = bidhb * swizzle + bidhb_residual;
                    let reverse: i32 = if SCHED == 3i32 { 0i32 } else { 1i32 };
                    (block, q_head_group_idx, reverse)
                }
            }
        };
        let block: i32 = sched.0;
        let q_head_group_idx: i32 = sched.1;
        if q_head_group_idx >= num_head_groups {
            return;
        }
        let q_m_idx: i32 = if sched.2 == 1i32 {
            num_q_blocks - 1i32 - block
        } else {
            block
        };
        let logical_tile_idx: i32 = q_m_idx * num_head_groups + q_head_group_idx;

        let lse_part: Partition<f32, { [1, NS_M] }> =
            lse_tv.partition_permuted(const_shape![1, NS_M], const_array![0, 1]);
        let lse_tile: Tile<f32, { [1, NS_M] }> = load_view_tko(
            &lse_part,
            [logical_tile_idx, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let lse_ns_m: Tile<f32, { [NUM_KV_SPLITS, M_EFF] }> =
            lse_tile.reshape(const_shape![NUM_KV_SPLITS, M_EFF]);
        let transpose_2d: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let lse_tile: Tile<f32, { [M_EFF, NUM_KV_SPLITS] }> = permute(lse_ns_m, transpose_2d);
        let lse_max: Tile<f32, { [M_EFF] }> = reduce_max(lse_tile, 1i32);
        let lse_max_col: Tile<f32, { [M_EFF, 1] }> = lse_max.reshape(const_shape![M_EFF, 1]);
        let lse_shifted: Tile<f32, { [M_EFF, NUM_KV_SPLITS] }> =
            lse_tile - lse_max_col.broadcast(const_shape![M_EFF, NUM_KV_SPLITS]);
        let scale_raw: Tile<f32, { [M_EFF, NUM_KV_SPLITS] }> = exp2(lse_shifted, ftz::Disabled);
        let scale_sum: Tile<f32, { [M_EFF] }> = reduce_sum(scale_raw, 1i32);
        let scale_sum_col: Tile<f32, { [M_EFF, 1] }> = scale_sum.reshape(const_shape![M_EFF, 1]);
        let eps: Tile<f32, { [M_EFF, 1] }> = constant(1.0e-8f32, const_shape![M_EFF, 1]);
        let scale_sum_safe: Tile<f32, { [M_EFF, 1] }> = max_tile(scale_sum_col, eps);
        let weights: Tile<f32, { [M_EFF, NUM_KV_SPLITS] }> = true_div(
            scale_raw,
            scale_sum_safe.broadcast(const_shape![M_EFF, NUM_KV_SPLITS]),
        );

        let att_part: Partition<f16, { [1, NS_M, CHUNK_D] }> =
            att_tv.partition_permuted(const_shape![1, NS_M, CHUNK_D], const_array![0, 1, 2]);
        let att_tile: Tile<f16, { [1, NS_M, CHUNK_D] }> = load_view_tko(
            &att_part,
            [logical_tile_idx, 0i32, d_chunk_id],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let att_ns_m_d: Tile<f16, { [NUM_KV_SPLITS, M_EFF, CHUNK_D] }> =
            att_tile.reshape(const_shape![NUM_KV_SPLITS, M_EFF, CHUNK_D]);
        let transpose_3d_01: Array<{ [1, 0, 2] }> = Array::<{ [1, 0, 2] }> {
            dims: &[1i32, 0i32, 2i32],
        };
        let att_m_ns_d: Tile<f16, { [M_EFF, NUM_KV_SPLITS, CHUNK_D] }> =
            permute(att_ns_m_d, transpose_3d_01);
        let att_tile: Tile<f32, { [M_EFF, NUM_KV_SPLITS, CHUNK_D] }> = convert_tile(att_m_ns_d);
        let w_3d: Tile<f32, { [M_EFF, NUM_KV_SPLITS, 1] }> =
            weights.reshape(const_shape![M_EFF, NUM_KV_SPLITS, 1]);
        let weighted: Tile<f32, { [M_EFF, NUM_KV_SPLITS, CHUNK_D] }> =
            att_tile * w_3d.broadcast(const_shape![M_EFF, NUM_KV_SPLITS, CHUNK_D]);
        let out_tile: Tile<f32, { [M_EFF, CHUNK_D] }> = reduce_sum(weighted, 1i32);
        let out_f16: Tile<f16, { [BM, GROUP, CHUNK_D] }> =
            convert_tile(out_tile.reshape(const_shape![BM, GROUP, CHUNK_D]));
        let mut out_part: PartitionMut<f16, { [BM, GROUP, CHUNK_D] }> =
            unsafe { out_tv.partition_full_mut(const_shape![BM, GROUP, CHUNK_D]) };
        unsafe {
            out_part.store(out_f16, [q_m_idx, q_head_group_idx, d_chunk_id]);
        }
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=4, max_divisibility=16,),
                         sm_120 = (occupancy=4, max_divisibility=16,),
                       ))]
    unsafe fn flash_attn_causal_seq_dynpos_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, d]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // [q_len, q_heads, d]
        qk_scale: f32,
        query_group_size: i32,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let query_start: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        // Decode graph uses q_len=1, BM=1: kv_len is position+1.
        let kv_len: i32 = query_start + 1i32;

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn fmha_causal<
        const BM: i32, // Query sequence tile size.
        const BN: i32, // KV sequence tile size.
        const D: i32,  // Head dimension.
        const CAUSAL: i32,
        const EVEN_K: i32,
    >(
        q: &Tensor<f16, { [-1, -1, D] }>,      // (m, h, d)
        k: &Tensor<f16, { [-1, -1, D] }>,      // (hkv, n, d)
        v: &Tensor<f16, { [-1, -1, D] }>,      // (hkv, n, d)
        out: &mut Tensor<f16, { [BM, 1, D] }>, // (m, b*h, d)
        qk_scale: f16,
        query_group_size: i32,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let input_pos: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale_f32: f32 = convert_scalar(qk_scale);
        let qk_scale: Tile<f32, { [BM, BN] }> =
            broadcast_scalar(qk_scale_f32 / log2, const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let k_seqlen: i32 = get_shape_dim(k.shape(), 1i32);
        let m_end: i32 = input_pos + (q_m_idx + 1i32) * BM;
        let mut mask_start: i32 = k_seqlen / BN;
        let mut tc: i32 = ceil_div(k_seqlen, BN);
        if CAUSAL == 1i32 {
            mask_start = (input_pos + q_m_idx * BM) / BN;
            let k_seqlen_tiles = k_seqlen / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, k_seqlen), BN);
        }

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);

        let offs_m_iota: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m_iota = offs_m_iota.reshape(const_shape![BM, 1]);
        let offs_m: Tile<i32, { [BM, 1] }> =
            broadcast_scalar(q_m_idx * BM + input_pos, const_shape![BM, 1]) + offs_m_iota;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m.broadcast(const_shape![BM, BN]);
        let k_seqlen_tile: Tile<i32, { [BM, BN] }> = k_seqlen.broadcast(const_shape![BM, BN]);
        let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> =
            constant(0.0f32, const_shape![BM, BN]) - constant(1.0e30f32, const_shape![BM, BN]);

        for j in 0i32..tc {
            let k_tile: Tile<f16, { [BN, D] }> = k_part
                .load([kv_head_idx, j, 0i32])
                .reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            qk = mma(tq, k_tile_trans, qk);

            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, const_shape![BM, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [BM, BN] }> = constant(true, const_shape![BM, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, k_seqlen_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                qk = qk + select(mask, mask_true, mask_false);
            }

            qk = qk * qk_scale;
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![BM, D]);

            let v_tile: Tile<f16, { [BN, D] }> = v_part
                .load([kv_head_idx, j, 0i32])
                .reshape(const_shape![BN, D]);
            let p_f16: Tile<f16, { [BM, BN] }> = convert_tile(p);
            acc = mma(p_f16, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }

    // Split-K + GQA decode attention, ported from
    // TileGym/src/tilegym/ops/cutile/gemma_attention_decode.py (Python cutile).
    // One CTA per (kv_head, kv_split). All Q heads within a GQA group are
    // processed together (packed along the GROUP dim) so K/V tiles are loaded
    // once per CTA and reused across GROUP queries. Combined with split-K,
    // this eliminates the O(kv_len) per-query cost of fmha_causal for decode.
    //
    // Grid: (num_kv_heads, NUM_KV_SPLITS).  (batch=1 dropped from dims to
    // match grout's single-batch tensor layouts.)
    //
    // Shapes (NOTE: cutile caps partition rank at 3, so split and group dims
    // are flattened in the scratch tensors — split is "outer", group "inner"):
    //   q           [num_kv_heads, GROUP, D]                 f16
    //   k, v        [num_kv_heads, S_kv, D]                  f16 (grout's k_cache layout)
    //   att_out     [num_kv_heads, NUM_KV_SPLITS * GROUP, D] f16  (per-CTA: [1, GROUP, D])
    //   lse_out     [num_kv_heads, NUM_KV_SPLITS * GROUP]    f32  (per-CTA: [1, GROUP])
    //   position_start: [1] u32 — current decode position (kv_len - 1)
    //
    // For Qwen3: num_kv_heads=8, GROUP=4, D=128. With NUM_KV_SPLITS=4, grid
    // is (8, 4) = 32 CTAs. Each CTA does S_kv/4 KV tokens × 4 queries.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn fmha_decode_gqa_split<
        const GROUP: i32,
        const BN: i32,
        const D: i32,
        const NUM_KV_SPLITS: i32,
        const LATENCY: i32, // pipeline depth for K/V load_from_view; tune per arch
    >(
        // Whole-tensor views (K/V partitioned internally by tile-index j).
        q: &Tensor<f16, { [-1, GROUP, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        v: &Tensor<f16, { [-1, -1, D] }>,
        // Per-CTA output tiles. The scratch tensors are
        //   att: [kv_heads, NUM_KV_SPLITS * GROUP, D]  — each CTA gets [1, GROUP, D]
        //   lse: [kv_heads, NUM_KV_SPLITS * GROUP]     — each CTA gets [1, GROUP]
        att_out: &mut Tensor<f16, { [1, GROUP, D] }>,
        lse_out: &mut Tensor<f32, { [1, GROUP] }>,
        qk_scale: f16,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let kv_head_id = pid.0;
        let split_id = pid.1;

        // s_kv = position_start + 1 (number of valid KV tokens at this step).
        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let input_pos: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));
        let s_kv: i32 = input_pos + 1i32;

        // qk_scale is passed in natural-log scale (1/sqrt(d)); we convert to
        // log2 scale once so the inner loop can use exp2 directly.
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let ln2: f32 = tile_to_scalar(log(two));
        let qk_scale_f32: f32 = convert_scalar(qk_scale);
        let qk_scale_log2: Tile<f32, { [BN, GROUP] }> =
            broadcast_scalar(qk_scale_f32 / ln2, const_shape![BN, GROUP]);

        // Split range over KV tiles (in units of BN tokens).
        let k_seqlen_tiles: i32 = ceil_div(s_kv, BN);
        let tiles_per_split: i32 = ceil_div(k_seqlen_tiles, NUM_KV_SPLITS);
        let start_tile: i32 = split_id * tiles_per_split;
        let mut end_tile: i32 = start_tile + tiles_per_split;
        end_tile = min(end_tile, k_seqlen_tiles);

        // Accumulators. m_i is kept rank-2 [GROUP, 1] (to match the rank-2
        // shape cutile's reduce_max produces after reshape, which avoids
        // constant-fold mismatches between `Tile<…{[GROUP]}>` and
        // `Tile<…{[4]}>` in max_tile).
        let neg_inf: Tile<f32, { [GROUP, 1] }> =
            constant(0.0f32, const_shape![GROUP, 1]) - constant(1.0e30f32, const_shape![GROUP, 1]);
        let mut m_i: Tile<f32, { [GROUP, 1] }> = neg_inf;
        let mut l_i: Tile<f32, { [BN, GROUP] }> = constant(1.0f32, const_shape![BN, GROUP]);
        let mut acc: Tile<f32, { [D, GROUP] }> = constant(0.0f32, const_shape![D, GROUP]);

        // Load Q once: [1, GROUP, D] → [GROUP, D] → [D, GROUP] (transposed).
        let q_part: Partition<f16, { [1, GROUP, D] }> = q.partition(const_shape![1, GROUP, D]);
        let q_tile: Tile<f16, { [1, GROUP, D] }> = q_part.load([kv_head_id, 0i32, 0i32]);
        let q_tile: Tile<f16, { [GROUP, D] }> = q_tile.reshape(const_shape![GROUP, D]);
        let transpose_2d: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let q_trans: Tile<f16, { [D, GROUP] }> = permute(q_tile, transpose_2d);

        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_col: Tile<i32, { [BN, 1] }> = offs_n_tile.reshape(const_shape![BN, 1]);
        let offs_n_2d: Tile<i32, { [BN, GROUP] }> = offs_n_col.broadcast(const_shape![BN, GROUP]);

        let s_kv_tile: Tile<i32, { [BN, GROUP] }> = s_kv.broadcast(const_shape![BN, GROUP]);
        let mask_true: Tile<f32, { [BN, GROUP] }> = constant(0.0f32, const_shape![BN, GROUP]);
        let mask_false: Tile<f32, { [BN, GROUP] }> = constant(0.0f32, const_shape![BN, GROUP])
            - constant(1.0e30f32, const_shape![BN, GROUP]);

        for j in start_tile..end_tile {
            let k_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &k_part,
                [kv_head_id, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);

            // qk = k @ q_T → [BN, GROUP]
            let mut qk: Tile<f32, { [BN, GROUP] }> = constant(0.0f32, const_shape![BN, GROUP]);
            qk = mma(k_tile, q_trans, qk);

            // Mask out-of-range KV positions (only matters at the last tile).
            if j == k_seqlen_tiles - 1i32 {
                let j_base: Tile<i32, { [BN, GROUP] }> =
                    broadcast_scalar(j * BN, const_shape![BN, GROUP]);
                let kv_pos: Tile<i32, { [BN, GROUP] }> = j_base + offs_n_2d;
                let valid: Tile<bool, { [BN, GROUP] }> = lt_tile(kv_pos, s_kv_tile);
                qk = qk + select(valid, mask_true, mask_false);
            }

            // Convert to log2 scale. Transpose qk to [GROUP, BN] so we can
            // reduce along the last axis (cutile's reduce_max only cleanly
            // supports axis=last in the existing grout/cutile examples).
            qk = qk * qk_scale_log2;
            let qk_t: Tile<f32, { [GROUP, BN] }> = permute(qk, transpose_2d);
            let qk_max_raw: Tile<f32, { [GROUP] }> = reduce_max(qk_t, 1i32);
            let qk_max_col: Tile<f32, { [GROUP, 1] }> = qk_max_raw.reshape(const_shape![GROUP, 1]);
            let m_ij: Tile<f32, { [GROUP, 1] }> = max_tile(m_i, qk_max_col);
            let qk_shifted: Tile<f32, { [GROUP, BN] }> =
                qk_t - m_ij.broadcast(const_shape![GROUP, BN]);
            let p_t: Tile<f32, { [GROUP, BN] }> = exp2(qk_shifted, ftz::Disabled);
            // Transpose p back to [BN, GROUP] for the V-mma below.
            let p: Tile<f32, { [BN, GROUP] }> = permute(p_t, transpose_2d);

            let alpha: Tile<f32, { [GROUP, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            let alpha_row: Tile<f32, { [1, GROUP] }> = alpha.reshape(const_shape![1, GROUP]);
            // Rescale l_i by alpha, accumulate p.
            l_i = l_i * alpha_row.broadcast(const_shape![BN, GROUP]) + p;
            // Rescale acc by alpha.
            acc = acc * alpha_row.broadcast(const_shape![D, GROUP]);

            // V tile: load [BN, D], transpose to [D, BN] for MMA.
            let v_tile: Tile<f16, { [1, BN, D] }> = load_view_tko(
                &v_part,
                [kv_head_id, j, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            );
            let v_tile: Tile<f16, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            let v_trans: Tile<f16, { [D, BN] }> = permute(v_tile, transpose_2d);

            // acc[D, GROUP] += v_T[D, BN] @ p[BN, GROUP]
            // p is f32; cast to f16 to match v_trans dtype for mma.
            let p_f16: Tile<f16, { [BN, GROUP] }> = convert_tile(p);
            acc = mma(v_trans, p_f16, acc);
            m_i = m_ij;
        }

        // Finalize this split: normalize acc by sum(l_i across BN) and emit
        // LSE = m_i + log2(l_sum) for the merge. Transpose first so we
        // reduce along the last axis (cutile pattern). Keep shapes rank-2
        // so subsequent max_tile etc. see matching symbolic shapes.
        let l_i_t: Tile<f32, { [GROUP, BN] }> = permute(l_i, transpose_2d);
        let l_sum_raw: Tile<f32, { [GROUP] }> = reduce_sum(l_i_t, 1i32);
        let l_sum: Tile<f32, { [GROUP, 1] }> = l_sum_raw.reshape(const_shape![GROUP, 1]);
        let eps_g: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let l_sum_safe: Tile<f32, { [GROUP, 1] }> = max_tile(l_sum, eps_g);
        let l_row: Tile<f32, { [1, GROUP] }> = l_sum_safe.reshape(const_shape![1, GROUP]);
        let acc_norm: Tile<f32, { [D, GROUP] }> =
            true_div(acc, l_row.broadcast(const_shape![D, GROUP]));

        // Transpose acc back to [GROUP, D] and store this CTA's per-split tile.
        let acc_out_t: Tile<f32, { [GROUP, D] }> = permute(acc_norm, transpose_2d);
        let acc_out_f16: Tile<f16, { [GROUP, D] }> = convert_tile(acc_out_t);
        let acc_out_3d: Tile<f16, { [1, GROUP, D] }> =
            acc_out_f16.reshape(const_shape![1, GROUP, D]);
        att_out.store(acc_out_3d);

        // LSE in log2 base: m_i + log2(l_sum). Both rank-2 [GROUP, 1].
        let lse_col: Tile<f32, { [GROUP, 1] }> = m_i + log2(l_sum_safe);
        let lse_out_tile: Tile<f32, { [1, GROUP] }> = lse_col.reshape(const_shape![1, GROUP]);
        lse_out.store(lse_out_tile);
    }

    // Merge per-split (att_partial, lse) into the final decode output.
    // Grid: (batch, num_kv_heads). Each CTA processes GROUP × NUM_KV_SPLITS
    // and produces GROUP output rows of length D.
    //
    // Standard flash-attention merge in log2 space:
    //   lse_max = max_s(lse_s)
    //   w_s    = 2^(lse_s - lse_max) / sum_s 2^(lse_s - lse_max)
    //   out[d] = sum_s w_s * acc_s[d]
    //
    // Shapes:
    //   att_partial: [batch, num_kv_heads, GROUP, NUM_KV_SPLITS, D]
    //   lse_partial: [batch, num_kv_heads, GROUP, NUM_KV_SPLITS]
    //   out:         [batch, num_kv_heads, GROUP, D]
    // TileGym's splitk_reduce_kernel uses occupancy=4; match it.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=4, max_divisibility=16,),
                         sm_120 = (occupancy=4, max_divisibility=16,),
                       ))]
    unsafe fn splitk_reduce_merge<
        const GROUP: i32,
        const D: i32,
        const CHUNK_D: i32, // per-CTA D chunk; grid dim 2 = D / CHUNK_D
        const NUM_KV_SPLITS: i32,
        const NS_GROUP: i32, // NUM_KV_SPLITS * GROUP, passed explicitly
        const LATENCY: i32,  // pipeline depth for input load_from_view
    >(
        // Scratch tensors from the split pass, with split and group flattened
        // into a single dim:
        //   att_partial: [kv_heads, NS_GROUP, D]   — per-CTA: [1, NS_GROUP, CHUNK_D]
        //   lse_partial: [kv_heads, NS_GROUP]      — per-CTA: [1, NS_GROUP]
        //   out:         [kv_heads, GROUP, D]      — per-CTA: [1, GROUP, CHUNK_D]
        //
        // Grid = (kv_heads, 1, D/CHUNK_D). Each CTA produces one [GROUP,
        // CHUNK_D] output slice. Splitting D across CTAs expands the grid
        // from (kv_heads,) = 8 CTAs to (kv_heads × D/CHUNK_D) = up to 64+
        // CTAs, closing the SM-undersub gap on 64-SM Blackwell. LSE and
        // weights are recomputed per-CTA (trivially cheap: GROUP ×
        // NUM_KV_SPLITS = ~32 ops) vs. sharing across CTAs.
        att_partial: &Tensor<f16, { [-1, NS_GROUP, D] }>,
        lse_partial: &Tensor<f32, { [-1, NS_GROUP] }>,
        out: &mut Tensor<f16, { [1, GROUP, CHUNK_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let kv_head_id = pid.0;
        let d_chunk_id = pid.2;

        // Load this CTA's [1, NS_GROUP] LSE tile and reshape to [NUM_KV_SPLITS, GROUP].
        let lse_part: Partition<f32, { [1, NS_GROUP] }> =
            lse_partial.partition(const_shape![1, NS_GROUP]);
        let lse_tile: Tile<f32, { [1, NS_GROUP] }> = load_view_tko(
            &lse_part,
            [kv_head_id, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        // Layout: split-major within NS_GROUP (split * GROUP + g), so reshape
        // to [NUM_KV_SPLITS, GROUP] then transpose → [GROUP, NUM_KV_SPLITS]
        // to match downstream accumulation.
        let lse_ns_g: Tile<f32, { [NUM_KV_SPLITS, GROUP] }> =
            lse_tile.reshape(const_shape![NUM_KV_SPLITS, GROUP]);
        let transpose_2d: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let lse_tile: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> = permute(lse_ns_g, transpose_2d);

        // Compute per-split weight w_s normalized across splits.
        let lse_max: Tile<f32, { [GROUP] }> = reduce_max(lse_tile, 1i32);
        let lse_max_col: Tile<f32, { [GROUP, 1] }> = lse_max.reshape(const_shape![GROUP, 1]);
        let lse_shifted: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> =
            lse_tile - lse_max_col.broadcast(const_shape![GROUP, NUM_KV_SPLITS]);
        let scale_raw: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> = exp2(lse_shifted, ftz::Disabled);
        let scale_sum: Tile<f32, { [GROUP] }> = reduce_sum(scale_raw, 1i32);
        let scale_sum_col: Tile<f32, { [GROUP, 1] }> = scale_sum.reshape(const_shape![GROUP, 1]);
        let eps: Tile<f32, { [GROUP, 1] }> = constant(1.0e-8f32, const_shape![GROUP, 1]);
        let scale_sum_safe: Tile<f32, { [GROUP, 1] }> = max_tile(scale_sum_col, eps);
        let weights: Tile<f32, { [GROUP, NUM_KV_SPLITS] }> = true_div(
            scale_raw,
            scale_sum_safe.broadcast(const_shape![GROUP, NUM_KV_SPLITS]),
        );

        // Load this CTA's CHUNK_D slice of [1, NS_GROUP, CHUNK_D] and
        // reshape to [NUM_KV_SPLITS, GROUP, CHUNK_D], then transpose
        // first two dims to get [GROUP, NUM_KV_SPLITS, CHUNK_D].
        let att_part: Partition<f16, { [1, NS_GROUP, CHUNK_D] }> =
            att_partial.partition(const_shape![1, NS_GROUP, CHUNK_D]);
        let att_tile: Tile<f16, { [1, NS_GROUP, CHUNK_D] }> = load_view_tko(
            &att_part,
            [kv_head_id, 0i32, d_chunk_id],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        let att_ns_g_d: Tile<f16, { [NUM_KV_SPLITS, GROUP, CHUNK_D] }> =
            att_tile.reshape(const_shape![NUM_KV_SPLITS, GROUP, CHUNK_D]);
        let transpose_3d_01: Array<{ [1, 0, 2] }> = Array::<{ [1, 0, 2] }> {
            dims: &[1i32, 0i32, 2i32],
        };
        let att_g_ns_d: Tile<f16, { [GROUP, NUM_KV_SPLITS, CHUNK_D] }> =
            permute(att_ns_g_d, transpose_3d_01);
        let att_tile: Tile<f32, { [GROUP, NUM_KV_SPLITS, CHUNK_D] }> = convert_tile(att_g_ns_d);

        // Broadcast weights to match att_tile dims.
        let w_3d: Tile<f32, { [GROUP, NUM_KV_SPLITS, 1] }> =
            weights.reshape(const_shape![GROUP, NUM_KV_SPLITS, 1]);
        let weighted: Tile<f32, { [GROUP, NUM_KV_SPLITS, CHUNK_D] }> =
            att_tile * w_3d.broadcast(const_shape![GROUP, NUM_KV_SPLITS, CHUNK_D]);
        let out_tile: Tile<f32, { [GROUP, CHUNK_D] }> = reduce_sum(weighted, 1i32);

        let out_f16: Tile<f16, { [GROUP, CHUNK_D] }> = convert_tile(out_tile);
        let out_3d: Tile<f16, { [1, GROUP, CHUNK_D] }> =
            out_f16.reshape(const_shape![1, GROUP, CHUNK_D]);
        out.store(out_3d);
    }

    /// Persistent RMS norm kernel using raw pointers and grid-stride loop.
    ///
    /// Weight W is loaded once outside the loop and reused across all rows.
    /// Uses 2D tiles [1, BLOCK_SIZE] with grid-stride over rows.
    /// Matches the conversion-skill rms_norm_sp pattern (Rule 12: raw pointer mode).
    ///
    /// Enable with GROUT_PERSISTENT_RMS_NORM=1.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn rms_norm_persistent_f16<const N: i32, const BLOCK_SIZE: i32>(
        x_ptr: *mut f16,
        x_rows: i32,
        x_stride: i32,
        w_ptr: *mut f16,
        out_ptr: *mut f16,
        out_stride: i32,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;

        // Alignment hints
        let x_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(x_ptr) };
        let w_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(w_ptr) };
        let out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(out_ptr) };
        let x_stride: i32 = unsafe { assume_div_by::<_, 16>(x_stride) };
        let out_stride: i32 = unsafe { assume_div_by::<_, 16>(out_stride) };
        let x_rows: i32 = unsafe { assume_bounds_lower::<_, 0>(x_rows) };

        let tok: Token = new_token_unordered();

        // Build W tensor view and load ONCE (shared across all rows)
        let w_ptile: PointerTile<*mut f16, { [] }> = pointer_to_tile(w_ptr);
        let w_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[N] };
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let w_tv: Tensor<f16, { [-1] }> =
            unsafe { make_tensor_view(w_ptile, w_shape, w_strides, tok) };
        let w_part: Partition<f16, { [BLOCK_SIZE] }> =
            w_tv.partition_permuted(const_shape![BLOCK_SIZE], const_array![0]);

        // Build X and out tensor views
        let x_ptile: PointerTile<*mut f16, { [] }> = pointer_to_tile(x_ptr);
        let x_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[x_rows, N] };
        let x_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[x_stride] };
        let x_tv: Tensor<f16, { [-1, -1] }> =
            unsafe { make_tensor_view(x_ptile, x_shape, x_strides, tok) };
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            x_tv.partition_permuted(tile_shape, const_array![0, 1]);

        let out_ptile: PointerTile<*mut f16, { [] }> = pointer_to_tile(out_ptr);
        let out_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> { dims: &[x_rows, N] };
        let out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> {
            dims: &[out_stride],
        };
        let out_tv: Tensor<f16, { [-1, -1] }> =
            unsafe { make_tensor_view(out_ptile, out_shape, out_strides, tok) };

        let n_f32: f32 = convert_scalar(N);

        // Grid-stride loop over rows
        let pid: (i32, i32, i32) = get_tile_block_id();
        let grid: (i32, i32, i32) = get_num_tile_blocks();
        for row in (pid.0..x_rows).step_by(grid.0 as usize) {
            // Pass 1: compute RMS
            let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
            for j in 0i32..num_tiles {
                let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                    &x_part,
                    [row, j],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
                rms = rms + tx * tx;
            }
            let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
            let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
            let rms: f32 = tile_to_scalar(rms);
            let inv_rms: f32 = rms / n_f32 + eps;
            let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
            let inv_rms: f32 = tile_to_scalar(inv_rms);
            let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

            // Pass 2: normalize with pre-loaded weight
            let mut out_part_mut: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
                unsafe { out_tv.partition_full_mut(tile_shape) };
            for j in 0i32..num_tiles {
                let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                    &x_part,
                    [row, j],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let tw_1d: Tile<f16, { [BLOCK_SIZE] }> = load_view_tko(
                    &w_part,
                    [j],
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = tw_1d.reshape(tile_shape);
                let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
                let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
                let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
                let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
                unsafe {
                    store_view_tko_mut(
                        &mut out_part_mut,
                        tout_f16,
                        [row, j],
                        ordering::Weak,
                        scope::TileBlock,
                        None,
                        tma::Enabled,
                    )
                };
            }
        }
    }

    /// Fused add + RMS norm kernel.
    ///
    /// Computes:  combined = residual + x
    ///            out      = rms_norm(combined, w, eps)
    ///            residual_out = combined  (updated residual for next layer)
    ///
    /// This saves a full read+write pass over the hidden state compared to
    /// separate Add + RmsNorm ops.
    #[cutile::entry(print_ir=false,
                        unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn add_rms_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        residual: &Tensor<f16, { [-1, N] }>,
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        residual_out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        // Ceiling division so BLOCK_SIZE does not have to divide N — lets us
        // ablate BLOCK_SIZE over pow-2 values (512 is the tuned default per
        // cutile-benchmarks/benches/rmsnorm.rs). Overhang lanes mask to zero
        // on load and are dropped on store via tile IR.
        let num_tiles: i32 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let residual_part: Partition<f16, { [1, BLOCK_SIZE] }> = residual.partition(tile_shape);
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);

        // First pass: add residual + x, accumulate sum of squares for RMS.
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = residual_part.load([row, j]);
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            rms = rms + combined * combined;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        // Second pass: write normalized output and updated residual.
        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        let mut res_out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { residual_out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = residual_part.load([row, j]);
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            let normed: Tile<f32, { [1, BLOCK_SIZE] }> = combined * inv_rms * tw;
            let normed_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(normed);
            let combined_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(combined);
            unsafe {
                out_part.store(normed_f16, [0i32, j]);
                res_out_part.store(combined_f16, [0i32, j]);
            }
        }
    }

    /// Decode-specialized fused add + RMS norm for contiguous single-row buffers.
    ///
    /// The generic Tensor/Partition entry above is kept for StepGraph and
    /// prefill-shaped inputs. This variant mirrors the decode graph's actual
    /// layout: all inputs and outputs are contiguous 1 x N buffers, so we can
    /// build direct raw-pointer views with alignment hints.
    #[cutile::entry(print_ir=false,
                        unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn add_rms_norm_decode_raw_f16<const N: i32, const BLOCK_SIZE: i32>(
        residual_ptr: *mut f16,
        x_ptr: *mut f16,
        w_ptr: *mut f16,
        out_ptr: *mut f16,
        residual_out_ptr: *mut f16,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let residual_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(residual_ptr) };
        let x_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(x_ptr) };
        let w_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(w_ptr) };
        let out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(out_ptr) };
        let residual_out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(residual_out_ptr) };

        let tok: Token = new_token_unordered();
        let shape_2d: Shape<{ [-1, N] }> = Shape::<{ [-1, N] }> { dims: &[1i32] };
        let strides_2d: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[N] };

        let residual_tv: Tensor<f16, { [-1, N] }> =
            unsafe { make_tensor_view(pointer_to_tile(residual_ptr), shape_2d, strides_2d, tok) };
        let x_tv: Tensor<f16, { [-1, N] }> =
            unsafe { make_tensor_view(pointer_to_tile(x_ptr), shape_2d, strides_2d, tok) };
        let w_shape: Shape<{ [N] }> = const_shape![N];
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let w_tv: Tensor<f16, { [N] }> =
            unsafe { make_tensor_view(pointer_to_tile(w_ptr), w_shape, w_strides, tok) };
        let out_tv: Tensor<f16, { [-1, N] }> =
            unsafe { make_tensor_view(pointer_to_tile(out_ptr), shape_2d, strides_2d, tok) };
        let residual_out_tv: Tensor<f16, { [-1, N] }> = unsafe {
            make_tensor_view(pointer_to_tile(residual_out_ptr), shape_2d, strides_2d, tok)
        };

        let residual_part: Partition<f16, { [1, BLOCK_SIZE] }> = residual_tv.partition(tile_shape);
        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x_tv.partition(tile_shape);

        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &residual_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &x_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            rms = rms + combined * combined;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let n: f32 = convert_scalar(N);
        let inv_rms: Tile<f32, { [] }> = true_div(rms, scalar_to_tile(n)) + scalar_to_tile(eps);
        let inv_rms: Tile<f32, { [] }> = rsqrt(inv_rms, ftz::Enabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w_tv.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out_tv.partition_full_mut(tile_shape) };
        let mut res_out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { residual_out_tv.partition_full_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tr_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &residual_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = load_view_tko(
                &x_part,
                [0i32, j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tw_1d: Tile<f16, { [BLOCK_SIZE] }> = load_view_tko(
                &w_part,
                [j],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = tw_1d.reshape(tile_shape);
            let tr: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tr_f16);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let combined: Tile<f32, { [1, BLOCK_SIZE] }> = tr + tx;
            let normed: Tile<f32, { [1, BLOCK_SIZE] }> = combined * inv_rms * tw;
            let normed_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(normed);
            let combined_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(combined);
            unsafe {
                store_view_tko_mut(
                    &mut out_part,
                    normed_f16,
                    [0i32, j],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut res_out_part,
                    combined_f16,
                    [0i32, j],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        }
    }

    /// Fused Q+K RMS norm: normalizes both Q and K heads in one kernel launch.
    ///
    /// Output is a single [num_q_rows + num_kv_rows, N] tensor. The first
    /// num_q_rows rows are normalized Q (using q_weight), the remaining
    /// rows are normalized K (using k_weight). The caller slices the output.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (max_divisibility=8,),
                         sm_120 = (max_divisibility=8,),
                       ))]
    unsafe fn qk_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        q: &Tensor<f16, { [-1, N] }>,
        k: &Tensor<f16, { [-1, N] }>,
        q_weight: &Tensor<f16, { [N] }>,
        k_weight: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
        num_q_rows: i32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let is_q: bool = row < num_q_rows;
        let local_row: i32 = if is_q { row } else { row - num_q_rows };

        let q_part: Partition<f16, { [1, BLOCK_SIZE] }> = q.partition(tile_shape);
        let k_part: Partition<f16, { [1, BLOCK_SIZE] }> = k.partition(tile_shape);

        // Pass 1: compute RMS
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = if is_q {
                q_part.load([local_row, j])
            } else {
                k_part.load([local_row, j])
            };
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        // Pass 2: normalize with the appropriate weight vector
        let qw_part: Partition<f16, { [BLOCK_SIZE] }> =
            q_weight.partition(const_shape![BLOCK_SIZE]);
        let kw_part: Partition<f16, { [BLOCK_SIZE] }> =
            k_weight.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = if is_q {
                q_part.load([local_row, j])
            } else {
                k_part.load([local_row, j])
            };
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = if is_q {
                qw_part.load([j]).reshape(tile_shape)
            } else {
                kw_part.load([j]).reshape(tile_shape)
            };
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
            unsafe { out_part.store(tout_f16, [0i32, j]) };
        }
    }

    /// Fused Q+K RoPE: applies rotary position embedding to both Q and K in one launch.
    ///
    /// Output is a single [seqlen, num_q_heads + num_kv_heads, D] tensor.
    /// First num_q_heads heads are rotated Q, remaining are rotated K.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn qk_rope_dynpos_f16<const D: i32, const HALF_D: i32, const LATENCY: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position_start: &Tensor<u32, { [1] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
        num_q_heads: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        let is_q: bool = head_idx < num_q_heads;
        let local_head: i32 = if is_q {
            head_idx
        } else {
            head_idx - num_q_heads
        };

        // Load input from Q or K based on head index. Both halves go through
        // load_from_view with Some(LATENCY) so the compiler can pipeline the
        // two cp_async issues with the constant-table loads + cos/sin compute.
        let q_part: Partition<f16, { [1, 1, HALF_D] }> = q.partition(const_shape![1, 1, HALF_D]);
        let k_part: Partition<f16, { [1, 1, HALF_D] }> = k.partition(const_shape![1, 1, HALF_D]);

        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        };
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(LATENCY),
                tma::Enabled,
            )
        };
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        // Position and frequency
        let pos_part = position_start.partition(const_shape![1]);
        let base_pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let base_pos_t: Tile<i32, { [1] }> = bitcast(base_pos_t_u32);
        let base_pos: i32 = tile_to_scalar(base_pos_t.reshape(const_shape![]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = base_pos + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        // Apply rotation
        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }

    /// Prefill-specialized fusion for:
    ///   q_norm + q_rope -> q_out
    ///   k_norm + k_rope -> k_cache
    ///   v                -> v_cache
    ///
    /// Grid is (seq_len, num_q_heads + num_kv_heads, 1). Each CTA computes
    /// the row RMS/RoPE for both halves and stores both halves, avoiding the
    /// older half-CTA path that duplicated the same D=128 math.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn qk_norm_rope_kv_prefill_raw_f16<
        const D: i32,
        const HALF_D: i32,
        const MAX_SEQ: i32,
    >(
        q_ptr: *mut f16,
        k_ptr: *mut f16,
        v_ptr: *mut f16,
        q_weight_ptr: *mut f16,
        k_weight_ptr: *mut f16,
        inv_freq_ptr: *mut f32,
        q_out_ptr: *mut f16,
        k_cache_ptr: *mut f16,
        v_cache_ptr: *mut f16,
        eps: f32,
        position_start: i32,
        seq_len: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
    ) {
        let half_shape: Shape<{ [1, 1, HALF_D] }> = const_shape![1, 1, HALF_D];
        let half_shape_2d: Shape<{ [1, HALF_D] }> = const_shape![1, HALF_D];

        let q_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_ptr) };
        let k_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_ptr) };
        let v_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_ptr) };
        let q_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_weight_ptr) };
        let k_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_weight_ptr) };
        let q_out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_out_ptr) };
        let k_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_cache_ptr) };
        let v_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_cache_ptr) };
        let seq_len: i32 = unsafe { assume_bounds_lower::<_, 0>(seq_len) };
        let num_q_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_q_heads) };
        let num_kv_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_kv_heads) };

        let tok: Token = new_token_unordered();

        let q_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[seq_len, num_q_heads],
        };
        let q_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[num_q_heads * D, D],
        };
        let q_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_ptr), q_shape, q_strides, tok) };
        let q_out_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_out_ptr), q_shape, q_strides, tok) };

        let kv_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[seq_len, num_kv_heads],
        };
        let kv_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[num_kv_heads * D, D],
        };
        let k_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_ptr), kv_shape, kv_strides, tok) };
        let v_tv: Tensor<f16, { [-1, -1, D] }> =
            unsafe { make_tensor_view(pointer_to_tile(v_ptr), kv_shape, kv_strides, tok) };

        let cache_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[num_kv_heads, MAX_SEQ],
        };
        let cache_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[MAX_SEQ * D, D],
        };
        let k_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(k_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };
        let v_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(v_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };

        let w_shape: Shape<{ [D] }> = const_shape![D];
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let q_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_weight_ptr), w_shape, w_strides, tok) };
        let k_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_weight_ptr), w_shape, w_strides, tok) };
        let inv_shape: Shape<{ [HALF_D] }> = const_shape![HALF_D];
        let inv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let inv_freq_tv: Tensor<f32, { [HALF_D] }> =
            unsafe { make_tensor_view(pointer_to_tile(inv_freq_ptr), inv_shape, inv_strides, tok) };

        let q_part: Partition<f16, { [1, 1, HALF_D] }> =
            q_tv.partition_permuted(const_shape![1, 1, HALF_D], const_array![0, 1, 2]);
        let k_part: Partition<f16, { [1, 1, HALF_D] }> =
            k_tv.partition_permuted(const_shape![1, 1, HALF_D], const_array![0, 1, 2]);
        let v_part: Partition<f16, { [1, 1, HALF_D] }> =
            v_tv.partition_permuted(const_shape![1, 1, HALF_D], const_array![0, 1, 2]);
        let q_weight_part: Partition<f16, { [HALF_D] }> =
            q_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);
        let k_weight_part: Partition<f16, { [HALF_D] }> =
            k_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);
        let inv_part: Partition<f32, { [HALF_D] }> =
            inv_freq_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let mut q_out_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { q_out_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };
        let mut k_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { k_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };
        let mut v_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { v_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };

        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let is_q: bool = head_idx < num_q_heads;
        let local_head: i32 = if is_q {
            head_idx
        } else {
            head_idx - num_q_heads
        };

        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = if is_q {
            load_view_tko(
                &q_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let x_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(x_lo_f16.reshape(half_shape_2d));
        let x_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(x_hi_f16.reshape(half_shape_2d));

        let rms_vec: Tile<f32, { [1, HALF_D] }> = x_lo * x_lo + x_hi * x_hi;
        let rms: Tile<f32, { [1] }> = reduce_sum(rms_vec, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let n: f32 = convert_scalar(D);
        let inv_rms: Tile<f32, { [] }> = true_div(rms, scalar_to_tile(n)) + scalar_to_tile(eps);
        let inv_rms: Tile<f32, { [] }> = rsqrt(inv_rms, ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, HALF_D] }> = inv_rms.broadcast(half_shape_2d);

        let w_lo_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_hi_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(w_lo_f16.reshape(half_shape_2d));
        let w_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(w_hi_f16.reshape(half_shape_2d));

        let norm_lo: Tile<f32, { [1, HALF_D] }> = x_lo * inv_rms * w_lo;
        let norm_hi: Tile<f32, { [1, HALF_D] }> = x_hi * inv_rms * w_hi;

        let freq: Tile<f32, { [HALF_D] }> = load_view_tko(
            &inv_part,
            [0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let pos_i: i32 = position_start + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [1, HALF_D] }> = (pos * freq).reshape(half_shape_2d);
        let cos_t: Tile<f32, { [1, HALF_D] }> = cos(theta);
        let sin_t: Tile<f32, { [1, HALF_D] }> = sin(theta);

        let y_lo: Tile<f32, { [1, HALF_D] }> = norm_lo * cos_t - norm_hi * sin_t;
        let y_hi: Tile<f32, { [1, HALF_D] }> = norm_hi * cos_t + norm_lo * sin_t;
        let y_lo_f16_2d: Tile<f16, { [1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16_2d: Tile<f16, { [1, HALF_D] }> = convert_tile(y_hi);
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = y_lo_f16_2d.reshape(half_shape);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = y_hi_f16_2d.reshape(half_shape);

        if is_q {
            unsafe {
                store_view_tko_mut(
                    &mut q_out_part,
                    y_lo_f16,
                    [seq_idx, local_head, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut q_out_part,
                    y_hi_f16,
                    [seq_idx, local_head, 1i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        } else {
            let v_lo: Tile<f16, { [1, 1, HALF_D] }> = load_view_tko(
                &v_part,
                [seq_idx, local_head, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let v_hi: Tile<f16, { [1, 1, HALF_D] }> = load_view_tko(
                &v_part,
                [seq_idx, local_head, 1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let cache_pos: i32 = position_start + seq_idx;
            unsafe {
                store_view_tko_mut(
                    &mut k_cache_part,
                    y_lo_f16,
                    [local_head, cache_pos, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut k_cache_part,
                    y_hi_f16,
                    [local_head, cache_pos, 1i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut v_cache_part,
                    v_lo,
                    [local_head, cache_pos, 0i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut v_cache_part,
                    v_hi,
                    [local_head, cache_pos, 1i32],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        }
    }

    /// Decode-specialized fusion for:
    ///   q_norm + q_rope -> q_out
    ///   k_norm + k_rope -> k_cache[position]
    ///   v                -> v_cache[position]
    ///
    /// Input is the contiguous QKV GEMV output:
    ///   [Q(num_q_heads * D), K(num_kv_heads * D), V(num_kv_heads * D)].
    /// Grid is (num_q_heads + num_kv_heads, 2, 1). Only rotated Q is written
    /// to q_out because decode attention reads Q directly from the front of
    /// q_out; rotated K is written straight to cache.
    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = (
                         sm_100 = (occupancy=1, max_divisibility=16,),
                         sm_120 = (occupancy=1, max_divisibility=16,),
                       ))]
    unsafe fn qk_norm_rope_kv_decode_raw_f16<
        const D: i32,
        const HALF_D: i32,
        const MAX_SEQ: i32,
    >(
        qkv_ptr: *mut f16,
        q_weight_ptr: *mut f16,
        k_weight_ptr: *mut f16,
        inv_freq_ptr: *mut f32,
        q_out_ptr: *mut f16,
        k_cache_ptr: *mut f16,
        v_cache_ptr: *mut f16,
        position_start: &Tensor<u32, { [1] }>,
        eps: f32,
        num_q_heads: i32,
        num_kv_heads: i32,
    ) {
        let half_shape_2d: Shape<{ [1, HALF_D] }> = const_shape![1, HALF_D];

        let qkv_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(qkv_ptr) };
        let q_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_weight_ptr) };
        let k_weight_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_weight_ptr) };
        let q_out_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(q_out_ptr) };
        let k_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(k_cache_ptr) };
        let v_cache_ptr: *mut f16 = unsafe { assume_div_by::<_, 16>(v_cache_ptr) };
        let num_q_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_q_heads) };
        let num_kv_heads: i32 = unsafe { assume_bounds_lower::<_, 0>(num_kv_heads) };
        let total_heads: i32 = num_q_heads + num_kv_heads;
        let qkv_elems: i32 = (num_q_heads + 2i32 * num_kv_heads) * D;

        let tok: Token = new_token_unordered();

        let qkv_shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[qkv_elems] };
        let qkv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let qkv_tv: Tensor<f16, { [-1] }> =
            unsafe { make_tensor_view(pointer_to_tile(qkv_ptr), qkv_shape, qkv_strides, tok) };
        let qkv_part: Partition<f16, { [HALF_D] }> =
            qkv_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let q_out_shape: Shape<{ [-1, D] }> = Shape::<{ [-1, D] }> {
            dims: &[total_heads],
        };
        let q_out_strides: Array<{ [-1, 1] }> = Array::<{ [-1, 1] }> { dims: &[D] };
        let q_out_tv: Tensor<f16, { [-1, D] }> = unsafe {
            make_tensor_view(pointer_to_tile(q_out_ptr), q_out_shape, q_out_strides, tok)
        };
        let mut q_out_part: PartitionMut<f16, { [1, HALF_D] }> =
            unsafe { q_out_tv.partition_full_mut(const_shape![1, HALF_D]) };

        let cache_shape: Shape<{ [-1, -1, D] }> = Shape::<{ [-1, -1, D] }> {
            dims: &[num_kv_heads, MAX_SEQ],
        };
        let cache_strides: Array<{ [-1, -1, 1] }> = Array::<{ [-1, -1, 1] }> {
            dims: &[MAX_SEQ * D, D],
        };
        let k_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(k_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };
        let v_cache_tv: Tensor<f16, { [-1, -1, D] }> = unsafe {
            make_tensor_view(
                pointer_to_tile(v_cache_ptr),
                cache_shape,
                cache_strides,
                tok,
            )
        };
        let mut k_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { k_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };
        let mut v_cache_part: PartitionMut<f16, { [1, 1, HALF_D] }> =
            unsafe { v_cache_tv.partition_full_mut(const_shape![1, 1, HALF_D]) };

        let w_shape: Shape<{ [D] }> = const_shape![D];
        let w_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let q_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(q_weight_ptr), w_shape, w_strides, tok) };
        let k_weight_tv: Tensor<f16, { [D] }> =
            unsafe { make_tensor_view(pointer_to_tile(k_weight_ptr), w_shape, w_strides, tok) };
        let q_weight_part: Partition<f16, { [HALF_D] }> =
            q_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);
        let k_weight_part: Partition<f16, { [HALF_D] }> =
            k_weight_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let inv_shape: Shape<{ [HALF_D] }> = const_shape![HALF_D];
        let inv_strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[] };
        let inv_freq_tv: Tensor<f32, { [HALF_D] }> =
            unsafe { make_tensor_view(pointer_to_tile(inv_freq_ptr), inv_shape, inv_strides, tok) };
        let inv_part: Partition<f32, { [HALF_D] }> =
            inv_freq_tv.partition_permuted(const_shape![HALF_D], const_array![0]);

        let pid: (i32, i32, i32) = get_tile_block_id();
        let head_idx = pid.0;
        let half_idx = pid.1;
        let is_q: bool = head_idx < num_q_heads;
        let local_head: i32 = if is_q {
            head_idx
        } else {
            head_idx - num_q_heads
        };
        let q_base_block: i32 = local_head * 2i32;
        let k_base_block: i32 = num_q_heads * 2i32 + local_head * 2i32;
        let v_base_block: i32 = num_q_heads * 2i32 + num_kv_heads * 2i32 + local_head * 2i32;
        let x_base_block: i32 = if is_q { q_base_block } else { k_base_block };

        let x_lo_f16: Tile<f16, { [HALF_D] }> = load_view_tko(
            &qkv_part,
            [x_base_block],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let x_hi_f16: Tile<f16, { [HALF_D] }> = load_view_tko(
            &qkv_part,
            [x_base_block + 1i32],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let x_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(x_lo_f16.reshape(half_shape_2d));
        let x_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(x_hi_f16.reshape(half_shape_2d));

        let rms_vec: Tile<f32, { [1, HALF_D] }> = x_lo * x_lo + x_hi * x_hi;
        let rms: Tile<f32, { [1] }> = reduce_sum(rms_vec, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let n: f32 = convert_scalar(D);
        let inv_rms: Tile<f32, { [] }> = true_div(rms, scalar_to_tile(n)) + scalar_to_tile(eps);
        let inv_rms: Tile<f32, { [] }> = rsqrt(inv_rms, ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, HALF_D] }> = inv_rms.broadcast(half_shape_2d);

        let w_lo_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_hi_f16: Tile<f16, { [HALF_D] }> = if is_q {
            load_view_tko(
                &q_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        } else {
            load_view_tko(
                &k_weight_part,
                [1i32],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            )
        };
        let w_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(w_lo_f16.reshape(half_shape_2d));
        let w_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(w_hi_f16.reshape(half_shape_2d));
        let norm_lo: Tile<f32, { [1, HALF_D] }> = x_lo * inv_rms * w_lo;
        let norm_hi: Tile<f32, { [1, HALF_D] }> = x_hi * inv_rms * w_hi;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let cache_pos: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let freq: Tile<f32, { [HALF_D] }> = load_view_tko(
            &inv_part,
            [0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(1i32),
            tma::Disabled,
        );
        let pos: f32 = convert_scalar(cache_pos);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [1, HALF_D] }> = (pos * freq).reshape(half_shape_2d);
        let cos_t: Tile<f32, { [1, HALF_D] }> = cos(theta);
        let sin_t: Tile<f32, { [1, HALF_D] }> = sin(theta);

        let y_lo: Tile<f32, { [1, HALF_D] }> = norm_lo * cos_t - norm_hi * sin_t;
        let y_hi: Tile<f32, { [1, HALF_D] }> = norm_hi * cos_t + norm_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_hi);

        if is_q {
            if half_idx == 0i32 {
                unsafe {
                    store_view_tko_mut(
                        &mut q_out_part,
                        y_lo_f16,
                        [local_head, 0i32],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(1i32),
                        tma::Disabled,
                    );
                }
            } else {
                unsafe {
                    store_view_tko_mut(
                        &mut q_out_part,
                        y_hi_f16,
                        [local_head, 1i32],
                        ordering::Weak,
                        scope::TileBlock,
                        Some(1i32),
                        tma::Disabled,
                    );
                }
            }
        } else {
            let v_half_f16: Tile<f16, { [HALF_D] }> = load_view_tko(
                &qkv_part,
                [v_base_block + half_idx],
                ordering::Weak,
                scope::TileBlock,
                Some(1i32),
                tma::Disabled,
            );
            let v_half: Tile<f16, { [1, 1, HALF_D] }> =
                v_half_f16.reshape(const_shape![1, 1, HALF_D]);
            let k_half: Tile<f16, { [1, 1, HALF_D] }> = if half_idx == 0i32 {
                y_lo_f16.reshape(const_shape![1, 1, HALF_D])
            } else {
                y_hi_f16.reshape(const_shape![1, 1, HALF_D])
            };
            unsafe {
                store_view_tko_mut(
                    &mut k_cache_part,
                    k_half,
                    [local_head, cache_pos, half_idx],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
                store_view_tko_mut(
                    &mut v_cache_part,
                    v_half,
                    [local_head, cache_pos, half_idx],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(1i32),
                    tma::Disabled,
                );
            }
        }
    }
}

#[allow(unused_imports)]
pub use kernels::{
    add_2d_f16, add_rms_norm_decode_raw_f16, add_rms_norm_f16, add_vec_f16, argmax_blocks_f16,
    argmax_reduce_blocks_to_u32, embedding_batch_f16, embedding_f16, flash_attn_causal_f16,
    flash_attn_causal_seq_dynpos_f16, flash_attn_causal_seq_f16, flash_attn_f16, fmha_causal,
    fmha_decode_gqa_split, fmha_prefill_causal, fmha_prefill_gqa, fmha_prefill_gqa_lpt,
    fmha_prefill_gqa_lpt_split, gather_row_f16, gemm_f16, group_gemm_f16_nt_desc,
    kv_cache_update_f16, kv_cache_update_seq_dynpos_f16, kv_cache_update_seq_f16,
    lm_head_argmax_blocks_f16, prefill_splitk_reduce_merge, qk_norm_f16,
    qk_norm_rope_kv_decode_raw_f16, qk_norm_rope_kv_prefill_raw_f16, qk_rope_dynpos_f16,
    rms_norm_f16, rms_norm_persistent_f16, rope_f16, rope_seq_dynpos_f16, rope_seq_f16,
    silu_mul_2d_f16, silu_mul_vec_f16, splitk_reduce_merge,
};
