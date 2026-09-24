//! Text-only Qwen3 and Qwen3.5 inference with resident packed Q4 weights.
//! The original cuTile FP16 engine remains separate. CUDA kernels are compiled
//! locally with NVRTC; no Python or remote model code is used for inference.
use crate::{
    model::GenerationOutput,
    mtp::{MtpOptions, accepted_count, top1},
    quantization::{Manifest, decode},
    quantized_config::TextConfig,
};
use anyhow::{Context, Result, bail, ensure};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use cutile::core::bf16;
use memmap2::MmapOptions;
use rand::Rng;
use safetensors::SafeTensors;
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

// Each scalar temporary lives until launch, including casts passed by reference.
macro_rules! launch_args {
    ($builder:ident, $cfg:expr;) => { unsafe { $builder.launch($cfg) } };
    ($builder:ident, $cfg:expr; $head:expr $(,$tail:expr)*) => {{
        let argument = $head;
        $builder.arg(argument);
        launch_args!($builder, $cfg; $($tail),*)
    }};
}
macro_rules! launch {
    ($gpu:expr, $name:literal, $cfg:expr; $($arg:expr),* $(,)?) => {{
        let function = $gpu.module.load_function($name)?;
        let mut builder = $gpu.stream.launch_builder(&function);
        // All buffers, dimensions and allocation lengths are checked by callers.
        launch_args!(builder, $cfg; $($arg),*).with_context(|| $name)?;
    }};
}

fn grid(x: usize, y: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (x as u32, y as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn flat(n: usize) -> LaunchConfig {
    grid(n.div_ceil(256), 1, 256)
}

struct Gpu {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}
impl Gpu {
    fn new() -> Result<Self> {
        let context = CudaContext::new(0)?;
        let major = context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        let minor = context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )?;
        ensure!(
            major >= 8,
            "BF16 backend requires compute capability 8.0 or newer"
        );
        let ptx = compile_ptx_with_opts(
            include_str!("quantized.cu"),
            CompileOptions {
                options: vec![
                    format!("--gpu-architecture=compute_{major}{minor}"),
                    "--std=c++14".into(),
                ],
                fmad: Some(false),
                ..Default::default()
            },
        )
        .context("compiling packed Q4 kernels with NVRTC")?;
        let module = context.load_module(ptx)?;
        Ok(Self {
            stream: context.default_stream(),
            module,
        })
    }
    fn zeros(&self, n: usize) -> Result<CudaSlice<u16>> {
        Ok(self.stream.alloc_zeros(n)?)
    }
    fn linear(&self, w: &Weight, x: &CudaSlice<u16>, rows: usize) -> Result<CudaSlice<u16>> {
        ensure!(w.shape.len() == 2, "linear weight must be a matrix");
        let (m, k) = (w.shape[0], w.shape[1]);
        ensure!(x.len() == rows * k, "linear input shape mismatch");
        let mut y = self.zeros(rows * m)?;
        let (mi, ki) = (m as i32, k as i32);
        match &w.data {
            Data::Q4 {
                codes,
                scales,
                group,
            } => {
                let g = *group as i32;
                launch!(self,"qlinear",grid(m.div_ceil(4),rows,128); x,codes,scales,&mut y,&mi,&ki,&g);
            }
            Data::Dense(data) => {
                launch!(self,"dense_linear",grid(m.div_ceil(4),rows,128); x,data,&mut y,&mi,&ki);
            }
        }
        Ok(y)
    }
    fn norm(
        &self,
        x: &CudaSlice<u16>,
        w: &Weight,
        width: usize,
        eps: f32,
        centered: bool,
    ) -> Result<CudaSlice<u16>> {
        ensure!(
            x.len() % width == 0 && w.shape == [width],
            "norm shape mismatch"
        );
        let mut y = self.zeros(x.len())?;
        launch!(self,"rms",grid(x.len()/width,1,256); x,w.dense()?,&mut y,&(width as i32),&eps,&(centered as i32));
        Ok(y)
    }
    fn add(&self, x: &mut CudaSlice<u16>, y: &CudaSlice<u16>) -> Result<()> {
        ensure!(x.len() == y.len(), "residual shape mismatch");
        let n = x.len() as i32;
        launch!(self,"add",flat(n as usize); x,y,&n);
        Ok(())
    }
}

enum Data {
    Q4 {
        codes: CudaSlice<u8>,
        scales: CudaSlice<f32>,
        group: usize,
    },
    Dense(CudaSlice<f32>),
}
struct Weight {
    shape: Vec<usize>,
    data: Data,
}
impl Weight {
    fn dense(&self) -> Result<&CudaSlice<f32>> {
        match &self.data {
            Data::Dense(v) => Ok(v),
            _ => bail!("expected a high-precision parameter"),
        }
    }
}
enum State {
    Attention {
        k: CudaSlice<u16>,
        v: CudaSlice<u16>,
    },
    Delta {
        conv: CudaSlice<u16>,
        recurrent: CudaSlice<f32>,
    },
}
/// A rollback point retains recurrent/conv states, and a logical KV prefix.
/// Attention tails are overwritten; recurrent updates cannot be undone by length.
pub struct Snapshot {
    position: usize,
    states: Vec<Option<(CudaSlice<u16>, CudaSlice<f32>)>>,
}

pub struct QuantizedEngine {
    gpu: Gpu,
    cfg: TextConfig,
    weights: HashMap<String, Weight>,
    states: Vec<State>,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    position: usize,
    eos: Vec<u32>,
    sample: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    chat: bool,
    thinking: bool,
    profile: bool,
    pub resident_weight_bytes: usize,
    mtp_trained: bool,
    mtp_mask: Option<u32>,
    mimo_template: bool,
}

impl QuantizedEngine {
    pub fn load(dir: &Path, max_seq_len: Option<usize>) -> Result<Self> {
        let cfg = TextConfig::load(dir)?;
        let manifest = Manifest::load(dir)?;
        let max_seq_len = max_seq_len.unwrap_or(4096).min(cfg.max_position_embeddings);
        ensure!(
            (1..=8192).contains(&max_seq_len),
            "quantized attention currently supports --max-seq-len 1..8192"
        );
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let tokenizer_mask = tokenizer.token_to_id("<|mtp_special_token_0|>");
        let mut mtp_mask = if cfg.qwen35 { None } else { tokenizer_mask };
        if dir.join("training_run.json").is_file() {
            let run: serde_json::Value =
                serde_json::from_slice(&fs::read(dir.join("training_run.json"))?)?;
            if matches!(run["objective"].as_str(), Some("hard" | "soft"))
                && run["steps"].as_u64().unwrap_or(0) > 0
            {
                let id = run["mask_id"]
                    .as_u64()
                    .context("trained MTP checkpoint has no mask_id")?;
                ensure!(
                    id < (cfg.vocab_size as u64),
                    "trained MTP mask outside vocabulary"
                );
                mtp_mask = Some(id as u32);
            }
        }
        let mtp_trained = mtp_mask.is_some();
        let mimo_template = fs::read_to_string(dir.join("chat_template.jinja"))
            .map(|text| text.contains("<|mimo_audio_start|>"))
            .unwrap_or(false);
        // Validate all required shapes before GPU allocation.
        let shapes = cfg.shapes();
        for (name, shape) in &shapes {
            let spec = manifest
                .tensors
                .get(name)
                .with_context(|| format!("missing tensor {name}"))?;
            ensure!(
                &spec.shape == shape,
                "{name}: expected {shape:?}, found {:?}",
                spec.shape
            );
        }
        let gpu = Gpu::new()?;
        let mut weights = HashMap::new();
        let mut resident_weight_bytes = 0;
        for (name, shape) in shapes {
            let spec = &manifest.tensors[&name];
            let file = fs::File::open(dir.join(&spec.file))?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let st = SafeTensors::deserialize(&mmap)?;
            let v = st.tensor(&name)?;
            let data = if spec.quantized {
                ensure!(
                    shape.len() == 2
                        && v.dtype() == safetensors::Dtype::U8
                        && v.shape() == [shape[0], shape[1].div_ceil(2)],
                    "invalid packed tensor {name}"
                );
                let sv = st.tensor(&format!("{name}.grout_scales"))?;
                ensure!(
                    sv.dtype() == safetensors::Dtype::F32
                        && sv.shape() == [shape[0], shape[1].div_ceil(manifest.group_size), 2],
                    "invalid scales {name}"
                );
                let scales = decode(sv.dtype(), sv.data())?;
                ensure!(
                    scales.iter().all(|x| x.is_finite()),
                    "non-finite scales {name}"
                );
                resident_weight_bytes += v.data().len() + sv.data().len();
                Data::Q4 {
                    codes: gpu.stream.clone_htod(v.data())?,
                    scales: gpu.stream.clone_htod(&scales)?,
                    group: manifest.group_size,
                }
            } else {
                ensure!(v.shape() == shape, "invalid dense tensor {name}");
                let values = decode(v.dtype(), v.data())?;
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "non-finite parameter {name}"
                );
                resident_weight_bytes += values.len() * 4;
                Data::Dense(gpu.stream.clone_htod(&values)?)
            };
            weights.insert(name, Weight { shape, data });
        }
        let mut states = vec![];
        for kind in &cfg.layer_types {
            states.push(if kind == "full_attention" {
                let n = max_seq_len * cfg.num_key_value_heads * cfg.head_dim;
                State::Attention {
                    k: gpu.zeros(n)?,
                    v: gpu.zeros(n)?,
                }
            } else {
                let channels = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                State::Delta {
                    conv: gpu.zeros(channels * cfg.linear_conv_kernel_dim)?,
                    recurrent: gpu.stream.alloc_zeros(
                        cfg.linear_num_value_heads
                            * cfg.linear_key_head_dim
                            * cfg.linear_value_head_dim,
                    )?,
                }
            });
        }
        let gen_cfg = crate::config::GenerationConfig::from_model_dir(dir)?;
        let eos = gen_cfg
            .as_ref()
            .and_then(|g| g.eos_token_id.clone())
            .map(|e| e.into_vec())
            .unwrap_or_else(|| vec![cfg.eos_token_id]);
        let temperature = gen_cfg.as_ref().and_then(|g| g.temperature).unwrap_or(1.);
        let top_k = gen_cfg.as_ref().and_then(|g| g.top_k).unwrap_or(0);
        let top_p = gen_cfg.as_ref().and_then(|g| g.top_p).unwrap_or(1.);
        gpu.stream.synchronize()?;
        println!(
            "Loaded packed Q4 weights: {:.3} GiB; BF16 activations/KV, FP32 recurrent state",
            resident_weight_bytes as f64 / (1u64 << 30) as f64
        );
        Ok(Self {
            gpu,
            cfg,
            weights,
            states,
            tokenizer,
            max_seq_len,
            position: 0,
            eos,
            sample: false,
            temperature,
            top_k,
            top_p,
            chat: true,
            thinking: false,
            profile: false,
            resident_weight_bytes,
            mtp_trained,
            mtp_mask,
            mimo_template,
        })
    }

    pub fn set_sampling_enabled(&mut self, v: bool) {
        self.sample = v;
    }
    pub fn set_chat_template_enabled(&mut self, v: bool) {
        self.chat = v;
    }
    pub fn set_profile_enabled(&mut self, v: bool) {
        self.profile = v;
    }
    pub fn set_thinking_enabled(&mut self, v: bool) {
        self.thinking = v;
    }
    pub fn mtp_mask_id(&self) -> Result<u32> {
        self.mtp_mask
            .context("checkpoint has no supported MTP training metadata/mask token")
    }

    pub fn reset(&mut self) -> Result<()> {
        // Attention memory is excluded by the length and overwritten on reuse.
        for state in &mut self.states {
            if let State::Delta { conv, recurrent } = state {
                self.gpu.stream.memset_zeros(conv)?;
                self.gpu.stream.memset_zeros(recurrent)?;
            }
        }
        self.position = 0;
        Ok(())
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        Ok(Snapshot {
            position: self.position,
            states: self
                .states
                .iter()
                .map(|s| match s {
                    State::Attention { .. } => Ok(None),
                    State::Delta { conv, recurrent } => {
                        Ok(Some((conv.try_clone()?, recurrent.try_clone()?)))
                    }
                })
                .collect::<Result<_>>()?,
        })
    }
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<()> {
        ensure!(
            snapshot.position <= self.position && snapshot.states.len() == self.states.len(),
            "invalid rollback point"
        );
        for (state, saved) in self.states.iter_mut().zip(&snapshot.states) {
            match (state, saved) {
                (State::Delta { conv, recurrent }, Some((c, r))) => {
                    self.gpu.stream.memcpy_dtod(c, conv)?;
                    self.gpu.stream.memcpy_dtod(r, recurrent)?;
                }
                (State::Attention { .. }, None) => {}
                _ => bail!("rollback state type mismatch"),
            }
        }
        self.position = snapshot.position;
        Ok(())
    }

    /// Append real or temporary tokens and return the requested suffix logits.
    /// All projections process multiple rows; DeltaNet scans rows in order.
    pub fn forward(&mut self, tokens: &[u32], logit_rows: usize) -> Result<Vec<Vec<f32>>> {
        let c = &self.cfg;
        let g = &self.gpu;
        let n = tokens.len();
        ensure!(
            n > 0 && n <= 256 && logit_rows <= n,
            "forward requires 1..256 rows and a valid logit suffix"
        );
        ensure!(
            self.position + n <= self.max_seq_len,
            "context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|&id| (id as usize) < c.vocab_size),
            "token ID outside vocabulary"
        );
        let ids = g.stream.clone_htod(tokens)?;
        let mut x = g.zeros(n * c.hidden_size)?;
        let embed = &self.weights[&format!("{}.embed_tokens.weight", c.prefix)];
        match &embed.data {
            Data::Q4 {
                codes,
                scales,
                group,
            } => {
                launch!(g,"embedding",grid(c.hidden_size.div_ceil(256),n,256); &ids,codes,scales,&mut x,&(c.hidden_size as i32),&(*group as i32));
            }
            Data::Dense(w) => {
                launch!(g,"dense_embedding",grid(c.hidden_size.div_ceil(256),n,256); &ids,w,&mut x,&(c.hidden_size as i32));
            }
        }
        for (i, state) in self.states.iter_mut().enumerate() {
            let p = format!("{}.layers.{i}", c.prefix);
            let w = |name: &str| &self.weights[&format!("{p}.{name}")];
            let norm = g.norm(
                &x,
                w("input_layernorm.weight"),
                c.hidden_size,
                c.rms_norm_eps,
                c.qwen35,
            )?;
            let mixed = match state {
                State::Attention { k: kc, v: vc } => {
                    let query = g.linear(w("self_attn.q_proj.weight"), &norm, n)?;
                    let key = g.linear(w("self_attn.k_proj.weight"), &norm, n)?;
                    let value = g.linear(w("self_attn.v_proj.weight"), &norm, n)?;
                    let qwidth = c.num_attention_heads * c.head_dim;
                    let mut gate = g.zeros(if c.qwen35 { n * qwidth } else { 1 })?;
                    let query = if c.qwen35 {
                        let mut q = g.zeros(n * qwidth)?;
                        launch!(g,"split_gate",flat(n*qwidth); &query,&mut q,&mut gate,&(c.num_attention_heads as i32),&(c.head_dim as i32),&(n as i32));
                        q
                    } else {
                        query
                    };
                    let mut q = g.norm(
                        &query,
                        w("self_attn.q_norm.weight"),
                        c.head_dim,
                        c.rms_norm_eps,
                        c.qwen35,
                    )?;
                    let mut k = g.norm(
                        &key,
                        w("self_attn.k_norm.weight"),
                        c.head_dim,
                        c.rms_norm_eps,
                        c.qwen35,
                    )?;
                    for (buf, heads) in [
                        (&mut q, c.num_attention_heads),
                        (&mut k, c.num_key_value_heads),
                    ] {
                        launch!(g,"rope",flat(n*heads*c.rotary_dim/2); buf,&(heads as i32),&(c.head_dim as i32),&(c.rotary_dim as i32),&(self.position as i32),&c.rope_theta,&(n as i32));
                    }
                    let kwidth = c.num_key_value_heads * c.head_dim;
                    launch!(g,"cache_write",flat(n*kwidth); &k,&value,&mut *kc,&mut *vc,&(kwidth as i32),&(self.position as i32),&(n as i32));
                    let mut attended = g.zeros(n * qwidth)?;
                    let mut cfg = grid(c.num_attention_heads, n, 256);
                    cfg.shared_mem_bytes = ((256 + self.position + n) * 4) as u32;
                    launch!(g,"attention",cfg; &q,&*kc,&*vc,&gate,&mut attended,&(c.num_attention_heads as i32),&(c.num_key_value_heads as i32),&(c.head_dim as i32),&(self.position as i32),&(c.qwen35 as i32));
                    g.linear(w("self_attn.o_proj.weight"), &attended, n)?
                }
                State::Delta {
                    conv: history,
                    recurrent,
                } => {
                    let qkv = g.linear(w("linear_attn.in_proj_qkv.weight"), &norm, n)?;
                    let z = g.linear(w("linear_attn.in_proj_z.weight"), &norm, n)?;
                    let a = g.linear(w("linear_attn.in_proj_a.weight"), &norm, n)?;
                    let beta = g.linear(w("linear_attn.in_proj_b.weight"), &norm, n)?;
                    let channels = 2 * c.linear_num_key_heads * c.linear_key_head_dim
                        + c.linear_num_value_heads * c.linear_value_head_dim;
                    let mut convolved = g.zeros(n * channels)?;
                    launch!(g,"conv",flat(channels); &qkv,w("linear_attn.conv1d.weight").dense()?,&mut *history,&mut convolved,&(channels as i32),&(c.linear_conv_kernel_dim as i32),&(n as i32));
                    let mut mixed =
                        g.zeros(n * c.linear_num_value_heads * c.linear_value_head_dim)?;
                    launch!(g,"delta",grid(c.linear_num_value_heads,1,256); &convolved,&a,&beta,w("linear_attn.A_log").dense()?,w("linear_attn.dt_bias").dense()?,&mut *recurrent,&mut mixed,&(c.linear_num_key_heads as i32),&(c.linear_num_value_heads as i32),&(c.linear_key_head_dim as i32),&(c.linear_value_head_dim as i32),&(n as i32));
                    let mut gated = g.zeros(mixed.len())?;
                    launch!(g,"gated_norm",grid(n*c.linear_num_value_heads,1,256); &mixed,&z,w("linear_attn.norm.weight").dense()?,&mut gated,&(c.linear_value_head_dim as i32),&c.rms_norm_eps);
                    g.linear(w("linear_attn.out_proj.weight"), &gated, n)?
                }
            };
            g.add(&mut x, &mixed)?;
            let norm = g.norm(
                &x,
                w("post_attention_layernorm.weight"),
                c.hidden_size,
                c.rms_norm_eps,
                c.qwen35,
            )?;
            let gate = g.linear(w("mlp.gate_proj.weight"), &norm, n)?;
            let up = g.linear(w("mlp.up_proj.weight"), &norm, n)?;
            let mut activated = g.zeros(gate.len())?;
            launch!(g,"swiglu",flat(gate.len()); &gate,&up,&mut activated,&(gate.len() as i32));
            let down = g.linear(w("mlp.down_proj.weight"), &activated, n)?;
            g.add(&mut x, &down)?;
        }
        self.position += n;
        if logit_rows == 0 {
            return Ok(vec![]);
        }
        let suffix = g
            .stream
            .clone_dtod(&x.slice((n - logit_rows) * c.hidden_size..))?;
        let norm = g.norm(
            &suffix,
            &self.weights[&format!("{}.norm.weight", c.prefix)],
            c.hidden_size,
            c.rms_norm_eps,
            c.qwen35,
        )?;
        let head = if c.tie_word_embeddings {
            embed
        } else {
            &self.weights["lm_head.weight"]
        };
        let logits = g.linear(head, &norm, logit_rows)?;
        let host = g.stream.clone_dtoh(&logits)?;
        let result: Vec<Vec<f32>> = host
            .chunks(c.vocab_size)
            .map(|row| {
                row.iter()
                    .map(|&bits| bf16::from_bits(bits).to_f32())
                    .collect()
            })
            .collect();
        ensure!(
            result.iter().flatten().all(|x| x.is_finite()),
            "non-finite logits"
        );
        Ok(result)
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        // Exact single-user text rendering for the supplied templates. Structured
        // conversations/tools/media must be formatted externally with --raw-prompt.
        let rendered = if self.chat {
            if self.cfg.qwen35 {
                ensure!(
                    self.mimo_template,
                    "unrecognized Qwen3.5 chat template; format the prompt with that checkpoint's template and use --raw-prompt"
                );
                format!(
                    "<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n{}",
                    if self.thinking { "" } else { "<think></think>" }
                )
            } else {
                format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{}",
                    if self.mtp_trained || self.thinking {
                        ""
                    } else {
                        "<think>\n\n</think>\n\n"
                    }
                )
            }
        } else {
            prompt.to_owned()
        };
        Ok(self
            .tokenizer
            .encode(rendered, true)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
            .get_ids()
            .to_vec())
    }

    fn choose(&self, logits: &[f32]) -> Result<u32> {
        if !self.sample {
            return Ok(top1(logits)?.0);
        }
        ensure!(
            self.temperature.is_finite()
                && self.temperature > 0.
                && self.top_p.is_finite()
                && self.top_p > 0.
                && self.top_p <= 1.,
            "invalid sampling parameters"
        );
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        if self.top_k > 0 {
            ids.truncate(self.top_k);
        }
        let mx = logits[ids[0]];
        let mut weights: Vec<f32> = ids
            .iter()
            .map(|&id| ((logits[id] - mx) / self.temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        let mut cumulative = 0.;
        let keep = weights
            .iter()
            .position(|&w| {
                cumulative += w / total;
                cumulative >= self.top_p
            })
            .map(|i| i + 1)
            .unwrap_or(weights.len());
        weights.truncate(keep);
        ids.truncate(keep);
        let mut sample = rand::thread_rng().r#gen::<f32>() * weights.iter().sum::<f32>();
        for (&id, &w) in ids.iter().zip(&weights) {
            if sample < w {
                return Ok(id as u32);
            }
            sample -= w;
        }
        Ok(*ids.last().unwrap() as u32)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        mtp: Option<MtpOptions>,
    ) -> Result<GenerationOutput> {
        if let Some(options) = mtp {
            options.validate(self.cfg.vocab_size)?;
            ensure!(!self.sample, "MTP requires greedy readout");
            ensure!(
                self.mtp_trained,
                "checkpoint has no MTP mask token/training metadata; an arbitrary --mask-id cannot enable MTP"
            );
            ensure!(
                Some(options.mask_id) == self.mtp_mask,
                "mask ID differs from checkpoint's MTP training mask"
            );
            ensure!(options.k <= 256, "MTP k must be at most 256");
        }
        let ids = self.encode_prompt(prompt)?;
        ensure!(!ids.is_empty(), "empty prompt");
        ensure!(
            ids.len() + max_new_tokens <= self.max_seq_len,
            "prompt and output budget exceed --max-seq-len"
        );
        self.reset()?;
        let start = Instant::now();
        let mut prompt_elapsed = Duration::ZERO;
        let mut generated = vec![];
        let mut chunks = vec![];
        let mut pending = ids.clone();
        while generated.len() < max_new_tokens {
            let k = mtp
                .map(|o| o.k)
                .unwrap_or(1)
                .min(max_new_tokens - generated.len());
            // Commit real rows before branching temporary masks. This is also the
            // commit boundary for recurrent and convolution states.
            let count = pending.len().div_ceil(32);
            let mut logits = vec![];
            for (i, chunk) in pending.chunks(32).enumerate() {
                logits = self.forward(chunk, if i + 1 == count { 1 } else { 0 })?;
            }
            if k > 1 {
                let saved = self.snapshot()?;
                let masks = vec![mtp.unwrap().mask_id; k - 1];
                logits.extend(self.forward(&masks, k - 1)?);
                self.restore(&saved)?;
            }
            pending.clear();
            if let Some(options) = mtp {
                let predictions = logits.iter().map(|l| top1(l)).collect::<Result<Vec<_>>>()?;
                let accepted = accepted_count(
                    &predictions.iter().map(|p| p.1).collect::<Vec<_>>(),
                    options.strategy,
                )?;
                for &(token, _) in &predictions[..accepted] {
                    pending.push(token);
                    if self.eos.contains(&token) {
                        break;
                    }
                }
                chunks.push(pending.len());
            } else {
                pending.push(self.choose(&logits[0])?);
            }
            generated.extend_from_slice(&pending);
            if prompt_elapsed.is_zero() {
                prompt_elapsed = start.elapsed();
            }
            if pending.last().is_some_and(|id| self.eos.contains(id)) {
                break;
            }
        }
        let total_elapsed = start.elapsed();
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(GenerationOutput{text,prompt_tokens:ids.len(),generated_tokens:generated.len(),token_ids:generated,mtp_chunks:chunks,prompt_elapsed,decode_elapsed:total_elapsed.saturating_sub(prompt_elapsed),total_elapsed,
            profile_report:self.profile.then(||format!("Packed weights: {} bytes; BF16 activations/KV; FP32 reductions/recurrent state; eager CUDA kernels",self.resident_weight_bytes))})
    }
}
