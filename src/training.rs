//! Native Rust Qwen3 training using Candle autograd. The cuTile inference
//! engine remains independent; checkpoints use the same HF tensor names.
use crate::{config::Qwen3Config, mtp::MtpBatch};
use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Tensor, Var};
use candle_nn::{Optimizer, VarMap};
use std::{collections::HashMap, path::Path};

pub struct NativeQwen3 {
    cfg: Qwen3Config,
    weights: HashMap<String, Tensor>,
    pub vars: VarMap,
}

/// Common differentiable interface; architecture-specific state semantics stay
/// in each model. Tokenizer compatibility is also checked by the training CLI.
pub trait TrainModel {
    fn vocab_size(&self) -> usize;
    fn max_positions(&self) -> usize;
    fn vars(&self) -> &VarMap;
    fn forward(
        &self,
        tokens: &[u32],
        positions: &[u32],
        bias: &[f32],
        rows: &[usize],
    ) -> Result<Tensor>;
}

impl TrainModel for NativeQwen3 {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    fn max_positions(&self) -> usize {
        self.max_positions()
    }
    fn vars(&self) -> &VarMap {
        &self.vars
    }
    fn forward(&self, t: &[u32], p: &[u32], b: &[f32], r: &[usize]) -> Result<Tensor> {
        self.forward(t, p, b, r)
    }
}

pub enum NativeModel {
    Qwen3(NativeQwen3),
    Qwen35(crate::training_qwen35::NativeQwen35),
}
impl NativeModel {
    pub fn load(dir: &Path, trainable: bool) -> Result<Self> {
        ensure!(
            !dir.join("quantization.json").exists(),
            "full-parameter training needs the original floating-point checkpoint; quantize its export separately"
        );
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        ensure!(
            matches!(
                config["model_type"].as_str(),
                Some("qwen3" | "qwen3_5" | "qwen3_5_text")
            ),
            "unsupported training architecture"
        );
        if matches!(
            config["model_type"].as_str(),
            Some("qwen3_5" | "qwen3_5_text")
        ) {
            Ok(Self::Qwen35(crate::training_qwen35::NativeQwen35::load(
                dir, trainable,
            )?))
        } else {
            Ok(Self::Qwen3(NativeQwen3::load(dir, trainable)?))
        }
    }
    pub fn tiny(trainable: bool, qwen35: bool) -> Result<Self> {
        if qwen35 {
            Ok(Self::Qwen35(crate::training_qwen35::NativeQwen35::tiny(
                trainable,
            )?))
        } else {
            Ok(Self::Qwen3(NativeQwen3::tiny(trainable)?))
        }
    }
    pub fn export(&self, dir: &Path, source: Option<&Path>) -> Result<()> {
        match self {
            Self::Qwen3(m) => m.export(dir, source),
            Self::Qwen35(m) => m.export(dir, source),
        }
    }
    pub fn compatible_with(&self, other: &Self) -> bool {
        self.vocab_size() == other.vocab_size()
    }
}
impl TrainModel for NativeModel {
    fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.vocab_size(),
            Self::Qwen35(m) => m.vocab_size(),
        }
    }
    fn max_positions(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.max_positions(),
            Self::Qwen35(m) => m.max_positions(),
        }
    }
    fn vars(&self) -> &VarMap {
        match self {
            Self::Qwen3(m) => &m.vars,
            Self::Qwen35(m) => &m.vars,
        }
    }
    fn forward(&self, t: &[u32], p: &[u32], b: &[f32], r: &[usize]) -> Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(t, p, b, r),
            Self::Qwen35(m) => m.forward(t, p, b, r),
        }
    }
}

fn shapes(c: &Qwen3Config) -> Vec<(String, Vec<usize>)> {
    let mut result = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![c.vocab_size, c.hidden_size],
        ),
        ("model.norm.weight".into(), vec![c.hidden_size]),
    ];
    if !c.tie_word_embeddings {
        result.push(("lm_head.weight".into(), vec![c.vocab_size, c.hidden_size]));
    }
    for i in 0..c.num_hidden_layers {
        for (name, shape) in [
            ("input_layernorm", vec![c.hidden_size]),
            ("post_attention_layernorm", vec![c.hidden_size]),
            ("self_attn.q_norm", vec![c.head_dim]),
            ("self_attn.k_norm", vec![c.head_dim]),
            (
                "self_attn.q_proj",
                vec![c.num_attention_heads * c.head_dim, c.hidden_size],
            ),
            (
                "self_attn.k_proj",
                vec![c.num_key_value_heads * c.head_dim, c.hidden_size],
            ),
            (
                "self_attn.v_proj",
                vec![c.num_key_value_heads * c.head_dim, c.hidden_size],
            ),
            (
                "self_attn.o_proj",
                vec![c.hidden_size, c.num_attention_heads * c.head_dim],
            ),
            ("mlp.gate_proj", vec![c.intermediate_size, c.hidden_size]),
            ("mlp.up_proj", vec![c.intermediate_size, c.hidden_size]),
            ("mlp.down_proj", vec![c.hidden_size, c.intermediate_size]),
        ] {
            result.push((format!("model.layers.{i}.{name}.weight"), shape));
        }
    }
    result
}

impl NativeQwen3 {
    pub fn load(dir: &Path, trainable: bool) -> Result<Self> {
        let cfg = Qwen3Config::from_model_dir(dir)?;
        let index: crate::config::SafetensorsIndex =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut shards: Vec<_> = index.weight_map.values().map(|p| dir.join(p)).collect();
        shards.sort();
        shards.dedup();
        // The mappings live through loading, and files are never modified.
        let source = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&shards)? };
        Self::from_loader(cfg, trainable, |name, _| {
            Ok(source.load(name, &Device::Cpu)?.to_dtype(DType::F32)?)
        })
    }

    fn from_loader(
        cfg: Qwen3Config,
        trainable: bool,
        mut load: impl FnMut(&str, &[usize]) -> Result<Tensor>,
    ) -> Result<Self> {
        ensure!(
            !cfg.use_sliding_window && cfg.head_dim % 2 == 0,
            "requires full attention and even head_dim"
        );
        ensure!(
            cfg.num_key_value_heads > 0 && cfg.num_attention_heads % cfg.num_key_value_heads == 0,
            "invalid GQA heads"
        );
        let vars = VarMap::new();
        let mut weights = HashMap::new();
        for (name, shape) in shapes(&cfg) {
            let tensor = load(&name, &shape).with_context(|| format!("loading {name}"))?;
            ensure!(
                tensor.dims() == shape,
                "{name}: expected {shape:?}, got {:?}",
                tensor.dims()
            );
            let tensor = if trainable {
                let var = Var::from_tensor(&tensor)?;
                let t = var.as_tensor().clone();
                vars.data().lock().unwrap().insert(name.clone(), var);
                t
            } else {
                tensor.detach()
            };
            weights.insert(name, tensor);
        }
        Ok(Self { cfg, weights, vars })
    }

    pub fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    pub fn max_positions(&self) -> usize {
        self.cfg.max_position_embeddings
    }

    pub fn compatible_with(&self, teacher: &Self) -> bool {
        self.cfg.vocab_size == teacher.cfg.vocab_size
    }

    fn weight(&self, name: &str) -> &Tensor {
        &self.weights[name]
    }

    fn linear(&self, x: &Tensor, name: &str) -> candle_core::Result<Tensor> {
        x.matmul(&self.weight(name).t()?)
    }

    fn norm(&self, x: &Tensor, name: &str) -> candle_core::Result<Tensor> {
        // Primitive operations preserve the autograd graph, unlike some
        // inference-only fused RMSNorm/RoPE kernels.
        let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + self.cfg.rms_norm_eps as f64)?.sqrt()?;
        x.broadcast_div(&scale)?.broadcast_mul(self.weight(name))
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let half = self.cfg.head_dim / 2;
        let a = x.narrow(2, 0, half)?;
        let b = x.narrow(2, half, half)?;
        Tensor::cat(
            &[
                &(a.broadcast_mul(cos)? - b.broadcast_mul(sin)?)?,
                &(b.broadcast_mul(cos)? + a.broadcast_mul(sin)?)?,
            ],
            2,
        )
    }

    /// Explicit logical positions and an additive visibility mask support both
    /// ordinary NTP and the multi-region blocked-attention training layout.
    pub fn forward(
        &self,
        tokens: &[u32],
        positions: &[u32],
        bias: &[f32],
        rows: &[usize],
    ) -> Result<Tensor> {
        let n = tokens.len();
        ensure!(
            n > 0 && positions.len() == n && bias.len() == n * n && !rows.is_empty(),
            "invalid forward layout"
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < self.cfg.vocab_size),
            "token outside vocabulary"
        );
        ensure!(
            positions
                .iter()
                .all(|&p| (p as usize) < self.cfg.max_position_embeddings),
            "position outside context"
        );
        ensure!(rows.iter().all(|&r| r < n), "logit row outside input");
        let dev = &Device::Cpu;
        let ids = Tensor::new(tokens, dev)?;
        let mut hidden = self
            .weight("model.embed_tokens.weight")
            .index_select(&ids, 0)?;
        let mask = Tensor::from_vec(bias.to_vec(), (1, n, n), dev)?;
        let h = self.cfg.num_attention_heads;
        let kv = self.cfg.num_key_value_heads;
        let d = self.cfg.head_dim;
        let mut cos = Vec::with_capacity(n * d / 2);
        let mut sin = Vec::with_capacity(n * d / 2);
        for &position in positions {
            for j in 0..d / 2 {
                let angle = position as f32 / self.cfg.rope_theta.powf(2.0 * j as f32 / d as f32);
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos, (1, n, d / 2), dev)?;
        let sin = Tensor::from_vec(sin, (1, n, d / 2), dev)?;
        let groups = Tensor::new(
            (0..h).map(|i| (i / (h / kv)) as u32).collect::<Vec<_>>(),
            dev,
        )?;
        for layer in 0..self.cfg.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            let norm = self.norm(&hidden, &format!("{p}.input_layernorm.weight"))?;
            let q = self
                .linear(&norm, &format!("{p}.self_attn.q_proj.weight"))?
                .reshape((n, h, d))?
                .transpose(0, 1)?;
            let k = self
                .linear(&norm, &format!("{p}.self_attn.k_proj.weight"))?
                .reshape((n, kv, d))?
                .transpose(0, 1)?;
            let v = self
                .linear(&norm, &format!("{p}.self_attn.v_proj.weight"))?
                .reshape((n, kv, d))?
                .transpose(0, 1)?
                .contiguous()?
                .index_select(&groups, 0)?;
            let q = self
                .rope(
                    &self.norm(&q, &format!("{p}.self_attn.q_norm.weight"))?,
                    &cos,
                    &sin,
                )?
                .contiguous()?;
            let k = self
                .rope(
                    &self.norm(&k, &format!("{p}.self_attn.k_norm.weight"))?,
                    &cos,
                    &sin,
                )?
                .contiguous()?
                .index_select(&groups, 0)?;
            let scores =
                (q.matmul(&k.transpose(1, 2)?)? / (d as f64).sqrt())?.broadcast_add(&mask)?;
            let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let attention = probs
                .matmul(&v)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((n, h * d))?;
            hidden =
                (&hidden + self.linear(&attention, &format!("{p}.self_attn.o_proj.weight"))?)?;
            let norm = self.norm(&hidden, &format!("{p}.post_attention_layernorm.weight"))?;
            let gate =
                candle_nn::ops::silu(&self.linear(&norm, &format!("{p}.mlp.gate_proj.weight"))?)?;
            let up = self.linear(&norm, &format!("{p}.mlp.up_proj.weight"))?;
            hidden =
                (&hidden + self.linear(&(gate * up)?, &format!("{p}.mlp.down_proj.weight"))?)?;
        }
        let selected = Tensor::new(rows.iter().map(|&r| r as u32).collect::<Vec<_>>(), dev)?;
        let hidden = self
            .norm(&hidden, "model.norm.weight")?
            .index_select(&selected, 0)?;
        Ok(self.linear(
            &hidden,
            if self.cfg.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
        )?)
    }

    pub fn export(&self, output: &Path, metadata_source: Option<&Path>) -> Result<()> {
        ensure!(
            !self.vars.all_vars().is_empty(),
            "cannot export frozen model via VarMap"
        );
        ensure!(
            !output.exists(),
            "output already exists: {}",
            output.display()
        );
        std::fs::create_dir_all(output)?;
        self.vars.save(output.join("model.safetensors"))?;
        let map: HashMap<_, _> = self
            .weights
            .keys()
            .map(|name| (name.clone(), "model.safetensors"))
            .collect();
        std::fs::write(
            output.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"weight_map":map}))?,
        )?;
        let config = if let Some(source) = metadata_source {
            let mut cfg: serde_json::Value =
                serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
            // Export is standard Qwen3; native MTP flags select the generation loop.
            cfg["dtype"] = serde_json::json!("float32");
            cfg.as_object_mut().unwrap().remove("auto_map");
            for file in [
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template.jinja",
            ] {
                if source.join(file).exists() {
                    std::fs::copy(source.join(file), output.join(file))?;
                }
            }
            cfg
        } else {
            let mut cfg = serde_json::to_value(&self.cfg)?;
            cfg["model_type"] = serde_json::json!("qwen3");
            cfg["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
            cfg["dtype"] = serde_json::json!("float32");
            cfg
        };
        std::fs::write(
            output.join("config.json"),
            serde_json::to_vec_pretty(&config)?,
        )?;
        Ok(())
    }

    /// Deterministic, small full Qwen3 for actual forward/backward smoke tests.
    pub fn tiny(trainable: bool) -> Result<Self> {
        let cfg = Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.,
            max_position_embeddings: 128,
            tie_word_embeddings: true,
            use_sliding_window: false,
            eos_token_id: 31,
        };
        Self::from_loader(cfg, trainable, |name, shape| {
            let values = if shape.len() == 1 {
                vec![1.; shape[0]]
            } else {
                let seed: usize = name.bytes().map(|b| b as usize).sum();
                (0..shape.iter().product())
                    .map(|i| ((i * 17 + seed) as f32 * 0.13).sin() * 0.08)
                    .collect()
            };
            Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Supervision {
    Hard,
    Soft,
}

/// Reference implementation uses per-position teacher CE/KL, not a product
/// of probabilities used as a scalar weight on the student's own token loss.
pub fn distillation_loss(student: &Tensor, teacher: &Tensor, mode: Supervision) -> Result<Tensor> {
    let teacher = teacher.detach();
    Ok(match mode {
        Supervision::Hard => candle_nn::loss::cross_entropy(student, &teacher.argmax(D::Minus1)?)?,
        Supervision::Soft => {
            let t = candle_nn::ops::log_softmax(&teacher, D::Minus1)?;
            let s = candle_nn::ops::log_softmax(student, D::Minus1)?;
            (t.exp()? * (&t - s)?)?.sum(D::Minus1)?.mean_all()?
        }
    })
}

pub fn student_forced_loss<S: TrainModel, T: TrainModel>(
    student: &S,
    teacher: &T,
    batch: &MtpBatch,
    mode: Supervision,
) -> Result<Tensor> {
    ensure!(
        student.vocab_size() == teacher.vocab_size(),
        "teacher and student vocabularies differ"
    );
    let bias = batch.attention_bias();
    let logits = student.forward(
        &batch.tokens,
        &batch.positions,
        &bias,
        &batch.prediction_rows,
    )?;
    let proposals = logits.detach().argmax(D::Minus1)?.to_vec1::<u32>()?;
    let teacher_tokens = batch.teacher_tokens(&proposals)?;
    let targets = teacher.forward(
        &teacher_tokens,
        &batch.positions,
        &bias,
        &batch.prediction_rows,
    )?;
    distillation_loss(&logits, &targets, mode)
}

pub fn causal_bias(n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|q| (0..n).map(move |k| if k <= q { 0. } else { f32::NEG_INFINITY }))
        .collect()
}

pub fn ntp_loss(model: &impl TrainModel, tokens: &[u32]) -> Result<Tensor> {
    ensure!(tokens.len() >= 2, "NTP needs at least two tokens");
    let n = tokens.len() - 1;
    let logits = model.forward(
        &tokens[..n],
        &(0..n as u32).collect::<Vec<_>>(),
        &causal_bias(n),
        &(0..n).collect::<Vec<_>>(),
    )?;
    Ok(candle_nn::loss::cross_entropy(
        &logits,
        &Tensor::new(&tokens[1..], &Device::Cpu)?,
    )?)
}

pub enum NativeOptimizer {
    Sgd(candle_nn::SGD),
    AdamW(candle_nn::AdamW),
}
impl NativeOptimizer {
    pub fn new(model: &impl TrainModel, adam: bool, lr: f64) -> Result<Self> {
        ensure!(lr.is_finite() && lr > 0., "learning rate must be positive");
        Ok(if adam {
            Self::AdamW(candle_nn::AdamW::new_lr(model.vars().all_vars(), lr)?)
        } else {
            Self::Sgd(candle_nn::SGD::new(model.vars().all_vars(), lr)?)
        })
    }
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        ensure!(
            loss.to_scalar::<f32>()?.is_finite(),
            "non-finite training loss"
        );
        let grads = loss.backward()?;
        match self {
            Self::Sgd(o) => o.step(&grads)?,
            Self::AdamW(o) => o.step(&grads)?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "writes a tiny checkpoint and numeric oracle inputs under target/mtp-training-oracle"]
    fn export_training_oracle() -> Result<()> {
        let student = NativeQwen3::tiny(true)?;
        let teacher = NativeQwen3::tiny(false)?;
        let dir = Path::new("target/mtp-training-oracle");
        student.export(dir, None)?;
        let batch = MtpBatch::new(&[2, 3, 4, 5, 6], &[1, 3], 3, 30)?;
        let logits = student.forward(
            &batch.tokens,
            &batch.positions,
            &batch.attention_bias(),
            &batch.prediction_rows,
        )?;
        let loss = student_forced_loss(&student, &teacher, &batch, Supervision::Hard)?;
        let grads = loss.backward()?;
        let mut gradients = HashMap::new();
        for (name, var) in student.vars.data().lock().unwrap().iter() {
            gradients.insert(
                name.clone(),
                grads
                    .get(var)
                    .context("missing gradient")?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
        let mask: Vec<Vec<bool>> = (0..batch.tokens.len())
            .map(|q| {
                (0..batch.tokens.len())
                    .map(|key| batch.visible(q, key))
                    .collect()
            })
            .collect();
        std::fs::write(
            dir.join("oracle.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "tokens":batch.tokens,"positions":batch.positions,"rows":batch.prediction_rows,
                "mask_rows":batch.mask_rows,"visible":mask,"k":batch.k,
                "logits":logits.to_vec2::<f32>()?,"hard_loss":loss.to_scalar::<f32>()?,"gradients":gradients
            }))?,
        )?;
        Ok(())
    }
    #[test]
    fn blocked_forward_matches_independent_regions() -> Result<()> {
        let model = NativeQwen3::tiny(false)?;
        let original = [2, 3, 4, 5, 6, 7];
        let batch = MtpBatch::new(&original, &[1, 4], 3, 30)?;
        let packed = model
            .forward(
                &batch.tokens,
                &batch.positions,
                &batch.attention_bias(),
                &batch.prediction_rows,
            )?
            .to_vec2::<f32>()?;
        for (region, anchor) in [1, 4].into_iter().enumerate() {
            let mut input = original[..=anchor].to_vec();
            input.extend([30, 30]);
            let n = input.len();
            let rows = model
                .forward(
                    &input,
                    &(0..n as u32).collect::<Vec<_>>(),
                    &causal_bias(n),
                    &(n - 3..n).collect::<Vec<_>>(),
                )?
                .to_vec2::<f32>()?;
            for j in 0..3 {
                for v in 0..32 {
                    assert!((packed[region * 3 + j][v] - rows[j][v]).abs() < 1e-5);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn backward_updates_student_and_freezes_teacher() -> Result<()> {
        let student = NativeQwen3::tiny(true)?;
        let teacher = NativeQwen3::tiny(false)?;
        let batch = MtpBatch::new(&[2, 3, 4, 5, 6], &[1, 3], 3, 30)?;
        let teacher_before = teacher
            .weight("model.embed_tokens.weight")
            .to_vec2::<f32>()?;
        let student_before = student
            .weight("model.embed_tokens.weight")
            .to_vec2::<f32>()?;
        let loss = student_forced_loss(&student, &teacher, &batch, Supervision::Hard)?;
        let before = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        for var in student.vars.all_vars() {
            let grad = grads
                .get(&var)
                .context("missing trainable weight gradient")?;
            assert!(
                grad.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite())
            );
        }
        let mut optim = NativeOptimizer::new(&student, false, 0.02)?;
        optim.backward_step(&loss)?;
        let after = student_forced_loss(&student, &teacher, &batch, Supervision::Hard)?
            .to_scalar::<f32>()?;
        assert!(after < before, "{after} >= {before}");
        assert_ne!(
            student_before,
            student
                .weight("model.embed_tokens.weight")
                .to_vec2::<f32>()?
        );
        assert_eq!(
            teacher_before,
            teacher
                .weight("model.embed_tokens.weight")
                .to_vec2::<f32>()?
        );
        assert!(teacher.vars.all_vars().is_empty());
        Ok(())
    }
    #[test]
    fn soft_kl_and_ntp_have_gradients() -> Result<()> {
        let student = NativeQwen3::tiny(true)?;
        let teacher = NativeQwen3::tiny(false)?;
        let batch = MtpBatch::new(&[1, 2, 3, 4], &[1], 3, 30)?;
        let loss = student_forced_loss(&student, &teacher, &batch, Supervision::Soft)?;
        assert!(loss.to_scalar::<f32>()? >= -1e-6);
        assert!(loss.backward()?.get(&student.vars.all_vars()[0]).is_some());
        let loss = ntp_loss(&student, &[1, 2, 3, 4])?;
        NativeOptimizer::new(&student, true, 0.001)?.backward_step(&loss)?;
        Ok(())
    }
    #[test]
    fn exported_checkpoint_reloads_without_logit_changes() -> Result<()> {
        let model = NativeQwen3::tiny(true)?;
        let dir = std::env::temp_dir().join(format!(
            "grout-mtp-roundtrip-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        model.export(&dir, None)?;
        let loaded = NativeQwen3::load(&dir, false)?;
        let original = model
            .forward(&[1, 2, 3], &[0, 1, 2], &causal_bias(3), &[2])?
            .to_vec2::<f32>()?;
        let restored = loaded
            .forward(&[1, 2, 3], &[0, 1, 2], &causal_bias(3), &[2])?
            .to_vec2::<f32>()?;
        assert_eq!(original, restored);
        // Only files created by this test, at a unique path, are removed.
        for name in [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ] {
            std::fs::remove_file(dir.join(name))?;
        }
        std::fs::remove_dir(dir)?;
        Ok(())
    }
}
