//! Differentiable CPU Qwen3.5 text model, including branched DeltaNet and
//! convolution histories for the existing packed MTP visibility layout.
use crate::{quantized_config::TextConfig, training::TrainModel};
use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Tensor, Var};
use candle_nn::VarMap;
use std::{collections::HashMap, path::Path};

pub struct NativeQwen35 {
    cfg: TextConfig,
    config_json: serde_json::Value,
    weights: HashMap<String, Tensor>,
    pub vars: VarMap,
}

/// A recurrent state must inherit exactly one causal history. Packed MTP
/// visibility forms a tree: real tokens bypass masks; each mask branch inherits
/// its anchor's real history. Reject unsupported arbitrary attention masks.
fn predecessors(bias: &[f32], n: usize) -> Result<Vec<Option<usize>>> {
    ensure!(bias.len() == n * n, "invalid visibility shape");
    let mut parents = vec![];
    for row in 0..n {
        for col in 0..n {
            ensure!(
                bias[row * n + col] == 0. || bias[row * n + col] == f32::NEG_INFINITY,
                "recurrent visibility must be boolean"
            );
            ensure!(
                col <= row || bias[row * n + col] == f32::NEG_INFINITY,
                "noncausal recurrent visibility"
            );
        }
        ensure!(bias[row * n + row] == 0., "each token must see itself");
        let parent = (0..row).rev().find(|&j| bias[row * n + j] == 0.);
        if let Some(p) = parent {
            for j in 0..p {
                ensure!(
                    (bias[row * n + j] == 0.) == (bias[p * n + j] == 0.),
                    "visibility does not form a recurrent branch"
                );
            }
        }
        parents.push(parent);
    }
    Ok(parents)
}

impl NativeQwen35 {
    pub fn load(dir: &Path, trainable: bool) -> Result<Self> {
        ensure!(
            !dir.join("quantization.json").exists(),
            "training requires floating-point source weights"
        );
        let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let index: crate::config::SafetensorsIndex =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut files: Vec<_> = index.weight_map.values().map(|p| dir.join(p)).collect();
        files.sort();
        files.dedup();
        let source = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };
        Self::from_loader(config, trainable, |name, _| {
            Ok(source.load(name, &Device::Cpu)?.to_dtype(DType::F32)?)
        })
    }
    fn from_loader(
        config_json: serde_json::Value,
        trainable: bool,
        mut load: impl FnMut(&str, &[usize]) -> Result<Tensor>,
    ) -> Result<Self> {
        let cfg = TextConfig::from_value(&config_json)?;
        ensure!(cfg.qwen35, "expected Qwen3.5");
        let vars = VarMap::new();
        let mut weights = HashMap::new();
        for (name, shape) in cfg.shapes() {
            let tensor = load(&name, &shape).with_context(|| name.clone())?;
            ensure!(tensor.dims() == shape, "shape mismatch for {name}");
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
        Ok(Self {
            cfg,
            config_json,
            weights,
            vars,
        })
    }
    fn weight(&self, name: &str) -> &Tensor {
        &self.weights[name]
    }
    fn linear(&self, x: &Tensor, name: &str) -> candle_core::Result<Tensor> {
        x.matmul(&self.weight(name).t()?)
    }
    fn norm(&self, x: &Tensor, name: &str, centered: bool) -> candle_core::Result<Tensor> {
        let denom = (x.sqr()?.mean_keepdim(D::Minus1)? + self.cfg.rms_norm_eps as f64)?.sqrt()?;
        let w = if centered {
            (self.weight(name) + 1.)?
        } else {
            self.weight(name).clone()
        };
        x.broadcast_div(&denom)?.broadcast_mul(&w)
    }
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let half = self.cfg.rotary_dim / 2;
        let a = x.narrow(2, 0, half)?;
        let b = x.narrow(2, half, half)?;
        let rotated = Tensor::cat(
            &[
                &(a.broadcast_mul(cos)? - b.broadcast_mul(sin)?)?,
                &(b.broadcast_mul(cos)? + a.broadcast_mul(sin)?)?,
            ],
            2,
        )?;
        if self.cfg.rotary_dim < self.cfg.head_dim {
            Tensor::cat(
                &[
                    &rotated,
                    &x.narrow(
                        2,
                        self.cfg.rotary_dim,
                        self.cfg.head_dim - self.cfg.rotary_dim,
                    )?,
                ],
                2,
            )
        } else {
            Ok(rotated)
        }
    }
    fn delta(&self, x: &Tensor, p: &str, parents: &[Option<usize>]) -> Result<Tensor> {
        let c = &self.cfg;
        let dev = &Device::Cpu;
        let n = parents.len();
        let (kh, vh, kd, vd) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        );
        let channels = 2 * kh * kd + vh * vd;
        let kernel = c.linear_conv_kernel_dim;
        let qkv = self.linear(x, &format!("{p}.in_proj_qkv.weight"))?;
        let z = self
            .linear(x, &format!("{p}.in_proj_z.weight"))?
            .reshape((n, vh, vd))?;
        let a = self.linear(x, &format!("{p}.in_proj_a.weight"))?;
        let beta = candle_nn::ops::sigmoid(&self.linear(x, &format!("{p}.in_proj_b.weight"))?)?;
        let groups = Tensor::new(
            (0..vh).map(|h| (h / (vh / kh)) as u32).collect::<Vec<_>>(),
            dev,
        )?;
        let conv_weight = self
            .weight(&format!("{p}.conv1d.weight"))
            .reshape((channels, kernel))?
            .t()?
            .contiguous()?;
        let zero_state = Tensor::zeros((vh, kd, vd), DType::F32, dev)?;
        let zero_history = Tensor::zeros((kernel - 1, channels), DType::F32, dev)?;
        let mut states: Vec<Tensor> = vec![];
        let mut histories: Vec<Tensor> = vec![];
        let mut outputs = vec![];
        for (t, &parent) in parents.iter().enumerate() {
            let previous = parent.map(|i| &states[i]).unwrap_or(&zero_state);
            let history = parent.map(|i| &histories[i]).unwrap_or(&zero_history);
            let window = Tensor::cat(&[history, &qkv.narrow(0, t, 1)?], 0)?;
            histories.push(window.narrow(0, 1, kernel - 1)?);
            let mixed = candle_nn::ops::silu(&(&window * &conv_weight)?.sum(0)?)?;
            let q = mixed.narrow(0, 0, kh * kd)?.reshape((kh, kd))?;
            let k = mixed.narrow(0, kh * kd, kh * kd)?.reshape((kh, kd))?;
            let q = q
                .broadcast_div(&(q.sqr()?.sum_keepdim(1)? + 1e-6)?.sqrt()?)?
                .index_select(&groups, 0)?;
            let q = (q / (kd as f64).sqrt())?;
            let k = k
                .broadcast_div(&(k.sqr()?.sum_keepdim(1)? + 1e-6)?.sqrt()?)?
                .index_select(&groups, 0)?;
            let v = mixed.narrow(0, 2 * kh * kd, vh * vd)?.reshape((vh, vd))?;
            let step = (a.narrow(0, t, 1)?.reshape(vh)? + self.weight(&format!("{p}.dt_bias")))?;
            let softplus = (step.relu()? + (step.abs()?.neg()?.exp()? + 1.)?.log()?)?;
            let decay = (self.weight(&format!("{p}.A_log")).exp()? * softplus)?
                .neg()?
                .exp()?
                .reshape((vh, 1, 1))?;
            let state = previous.broadcast_mul(&decay)?;
            let memory = state.broadcast_mul(&k.unsqueeze(2)?)?.sum(1)?;
            let change = (v - memory)?.broadcast_mul(&beta.narrow(0, t, 1)?.reshape((vh, 1))?)?;
            let state = (&state + k.unsqueeze(2)?.broadcast_mul(&change.unsqueeze(1)?)?)?;
            let output = state.broadcast_mul(&q.unsqueeze(2)?)?.sum(1)?;
            outputs.push(
                self.norm(&output, &format!("{p}.norm.weight"), false)?
                    * candle_nn::ops::silu(&z.narrow(0, t, 1)?.reshape((vh, vd))?)?,
            );
            states.push(state);
        }
        let outputs = outputs
            .into_iter()
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mixed = Tensor::stack(&outputs, 0)?.reshape((n, vh * vd))?;
        Ok(self.linear(&mixed, &format!("{p}.out_proj.weight"))?)
    }

    pub fn export(&self, output: &Path, source: Option<&Path>) -> Result<()> {
        ensure!(
            !self.vars.all_vars().is_empty(),
            "cannot export frozen model"
        );
        ensure!(!output.exists(), "output already exists");
        std::fs::create_dir_all(output)?;
        self.vars.save(output.join("model.safetensors"))?;
        let map: HashMap<_, _> = self
            .weights
            .keys()
            .map(|n| (n.clone(), "model.safetensors"))
            .collect();
        std::fs::write(
            output.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"weight_map":map}))?,
        )?;
        let mut config = self.config_json.clone();
        config["dtype"] = serde_json::json!("float32");
        if config["text_config"].is_object() {
            config["text_config"]["dtype"] = serde_json::json!("float32");
        }
        config.as_object_mut().unwrap().remove("auto_map");
        // This trainer updates/exports text weights only, never a vision encoder.
        config["grout_text_only"] = serde_json::json!(true);
        std::fs::write(
            output.join("config.json"),
            serde_json::to_vec_pretty(&config)?,
        )?;
        if let Some(source) = source {
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
        }
        Ok(())
    }
    pub fn tiny(trainable: bool) -> Result<Self> {
        let config = serde_json::json!({"model_type":"qwen3_5_text","architectures":["Qwen3_5ForCausalLM"],"hidden_size":16,"intermediate_size":24,"vocab_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"rms_norm_eps":1e-6,"max_position_embeddings":128,"eos_token_id":31,"tie_word_embeddings":false,"layer_types":["linear_attention","full_attention"],"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4,"rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.5,"mrope_section":[1,0,0],"mrope_interleaved":true}});
        Self::from_loader(config, trainable, |name, shape| {
            let size = shape.iter().product();
            let offset = name.bytes().map(|b| b as usize).sum::<usize>();
            let values = (0..size)
                .map(|i| {
                    if name.ends_with("linear_attn.norm.weight") {
                        1.
                    } else if name.contains("layernorm")
                        || name.ends_with("model.norm.weight")
                        || name.ends_with("q_norm.weight")
                        || name.ends_with("k_norm.weight")
                    {
                        0.
                    } else if name.ends_with("A_log") || name.ends_with("dt_bias") {
                        0.1
                    } else {
                        ((i + offset) as f32 * 0.37).sin() * 0.07
                    }
                })
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
        })
    }
}

impl TrainModel for NativeQwen35 {
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn max_positions(&self) -> usize {
        self.cfg.max_position_embeddings
    }
    fn vars(&self) -> &VarMap {
        &self.vars
    }
    fn forward(
        &self,
        tokens: &[u32],
        positions: &[u32],
        bias: &[f32],
        rows: &[usize],
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let n = tokens.len();
        let dev = &Device::Cpu;
        ensure!(
            n > 0 && positions.len() == n && !rows.is_empty() && rows.iter().all(|&r| r < n),
            "invalid forward layout"
        );
        ensure!(
            tokens.iter().all(|&t| (t as usize) < c.vocab_size)
                && positions
                    .iter()
                    .all(|&p| (p as usize) < c.max_position_embeddings),
            "token/position outside configuration"
        );
        let parents = predecessors(bias, n)?;
        let mut x = self
            .weight(&format!("{}.embed_tokens.weight", c.prefix))
            .index_select(&Tensor::new(tokens, dev)?, 0)?;
        let mask = Tensor::from_vec(bias.to_vec(), (1, n, n), dev)?;
        let (h, kv, d) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let half = c.rotary_dim / 2;
        let mut cos = vec![];
        let mut sin = vec![];
        for &pos in positions {
            for j in 0..half {
                let angle = pos as f32 / c.rope_theta.powf(2. * j as f32 / c.rotary_dim as f32);
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos, (1, n, half), dev)?;
        let sin = Tensor::from_vec(sin, (1, n, half), dev)?;
        let groups = Tensor::new(
            (0..h).map(|i| (i / (h / kv)) as u32).collect::<Vec<_>>(),
            dev,
        )?;
        for (layer, kind) in c.layer_types.iter().enumerate() {
            let p = format!("{}.layers.{layer}", c.prefix);
            let norm = self.norm(&x, &format!("{p}.input_layernorm.weight"), true)?;
            let attention = if kind == "linear_attention" {
                self.delta(&norm, &format!("{p}.linear_attn"), &parents)?
            } else {
                let query = self
                    .linear(&norm, &format!("{p}.self_attn.q_proj.weight"))?
                    .reshape((n, h, 2 * d))?;
                let gate = query.narrow(2, d, d)?.contiguous()?.reshape((n, h * d))?;
                let q = query.narrow(2, 0, d)?.transpose(0, 1)?;
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
                        &self.norm(&q, &format!("{p}.self_attn.q_norm.weight"), true)?,
                        &cos,
                        &sin,
                    )?
                    .contiguous()?;
                let k = self
                    .rope(
                        &self.norm(&k, &format!("{p}.self_attn.k_norm.weight"), true)?,
                        &cos,
                        &sin,
                    )?
                    .contiguous()?
                    .index_select(&groups, 0)?;
                let scores =
                    (q.matmul(&k.transpose(1, 2)?)? / (d as f64).sqrt())?.broadcast_add(&mask)?;
                let output = candle_nn::ops::softmax(&scores, D::Minus1)?
                    .matmul(&v)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((n, h * d))?;
                self.linear(
                    &(output * candle_nn::ops::sigmoid(&gate)?)?,
                    &format!("{p}.self_attn.o_proj.weight"),
                )?
            };
            x = (x + attention)?;
            let norm = self.norm(&x, &format!("{p}.post_attention_layernorm.weight"), true)?;
            let gate =
                candle_nn::ops::silu(&self.linear(&norm, &format!("{p}.mlp.gate_proj.weight"))?)?;
            let up = self.linear(&norm, &format!("{p}.mlp.up_proj.weight"))?;
            x = (x + self.linear(&(gate * up)?, &format!("{p}.mlp.down_proj.weight"))?)?;
        }
        let selected = Tensor::new(rows.iter().map(|&r| r as u32).collect::<Vec<_>>(), dev)?;
        let x = self
            .norm(&x, &format!("{}.norm.weight", c.prefix), true)?
            .index_select(&selected, 0)?;
        Ok(self.linear(
            &x,
            &if c.tie_word_embeddings {
                format!("{}.embed_tokens.weight", c.prefix)
            } else {
                "lm_head.weight".into()
            },
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mtp::MtpBatch,
        training::{NativeOptimizer, Supervision, causal_bias, student_forced_loss},
    };

    #[test]
    fn masks_do_not_contaminate_real_recurrent_or_attention_history() -> Result<()> {
        let model = NativeQwen35::tiny(false)?;
        let tokens = [1, 2, 3, 4, 5];
        let mut batch = MtpBatch::new(&tokens, &[0, 2], 3, 30)?;
        let real: Vec<usize> = batch
            .regions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.is_none().then_some(i))
            .collect();
        let expected =
            model.forward(&tokens, &[0, 1, 2, 3, 4], &causal_bias(5), &[0, 1, 2, 3, 4])?;
        let packed = model.forward(
            &batch.tokens,
            &batch.positions,
            &batch.attention_bias(),
            &real,
        )?;
        let error = (&expected - &packed)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error < 1e-5, "real stream differs by {error}");
        for &row in &batch.mask_rows {
            batch.tokens[row] = 29;
        }
        let changed = model.forward(
            &batch.tokens,
            &batch.positions,
            &batch.attention_bias(),
            &real,
        )?;
        assert!((&changed - &packed)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        Ok(())
    }

    #[test]
    fn qwen35_full_parameter_distillation_has_finite_gradients_and_updates() -> Result<()> {
        let student = NativeQwen35::tiny(true)?;
        let teacher = NativeQwen35::tiny(false)?;
        let batch = MtpBatch::new(&[1, 2, 3, 4, 5], &[0, 2], 3, 30)?;
        let loss = student_forced_loss(&student, &teacher, &batch, Supervision::Hard)?;
        let before = loss.to_scalar::<f32>()?;
        let gradients = loss.backward()?;
        for (name, var) in student.vars.data().lock().unwrap().iter() {
            let gradient = gradients
                .get(var)
                .with_context(|| format!("missing gradient {name}"))?;
            assert!(
                gradient
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite()),
                "{name}"
            );
        }
        assert!(teacher.vars.all_vars().is_empty());
        let mut optimizer = NativeOptimizer::new(&student, false, 0.01)?;
        optimizer.backward_step(&loss)?;
        let after = student_forced_loss(&student, &teacher, &batch, Supervision::Hard)?
            .to_scalar::<f32>()?;
        assert!(after < before, "{before} -> {after}");
        let soft = student_forced_loss(&student, &teacher, &batch, Supervision::Soft)?;
        assert!(soft.to_scalar::<f32>()?.is_finite());
        Ok(())
    }

    #[test]
    #[ignore = "exports a small model, forward values and every gradient for the independent PyTorch oracle"]
    fn export_qwen35_training_oracle() -> Result<()> {
        let root = std::env::var("GROUT_QWEN35_ORACLE")
            .unwrap_or_else(|_| "target/qwen35-training-oracle".into());
        let root = Path::new(&root);
        let student = NativeQwen35::tiny(true)?;
        let teacher = NativeQwen35::tiny(false)?;
        student.export(root, None)?;
        let batch = MtpBatch::new(&[1, 2, 3, 4, 5], &[0, 2], 3, 30)?;
        let logits = student.forward(
            &batch.tokens,
            &batch.positions,
            &batch.attention_bias(),
            &batch.prediction_rows,
        )?;
        let proposals = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let teacher_tokens = batch.teacher_tokens(&proposals)?;
        let targets = teacher
            .forward(
                &teacher_tokens,
                &batch.positions,
                &batch.attention_bias(),
                &batch.prediction_rows,
            )?
            .argmax(D::Minus1)?
            .to_vec1::<u32>()?;
        let loss = student_forced_loss(&student, &teacher, &batch, Supervision::Hard)?;
        let grads = loss.backward()?;
        let mut gradients = HashMap::new();
        for (name, var) in student.vars.data().lock().unwrap().iter() {
            gradients.insert(
                name.clone(),
                grads
                    .get(var)
                    .context("gradient missing")?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
        let ntp = student
            .forward(
                &[1, 2, 3, 4, 5],
                &[0, 1, 2, 3, 4],
                &causal_bias(5),
                &[0, 1, 2, 3, 4],
            )?
            .to_vec2::<f32>()?;
        let visible: Vec<Vec<bool>> = (0..batch.tokens.len())
            .map(|q| {
                (0..batch.tokens.len())
                    .map(|k| batch.visible(q, k))
                    .collect()
            })
            .collect();
        std::fs::write(
            root.join("oracle.json"),
            serde_json::to_vec(&serde_json::json!({
                "tokens":batch.tokens,"positions":batch.positions,"rows":batch.prediction_rows,"visible":visible,
                "teacher_tokens":teacher_tokens,"targets":targets,"logits":logits.to_vec2::<f32>()?,"hard_loss":loss.to_scalar::<f32>()?,"gradients":gradients,"ntp_logits":ntp
            }))?,
        )?;
        let restored = NativeQwen35::load(root, false)?;
        let restored_logits = restored.forward(
            &batch.tokens,
            &batch.positions,
            &batch.attention_bias(),
            &batch.prediction_rows,
        )?;
        assert_eq!(logits.to_vec2::<f32>()?, restored_logits.to_vec2::<f32>()?);
        Ok(())
    }
}
