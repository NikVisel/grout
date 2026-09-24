use anyhow::{Result, ensure};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    #[serde(skip)]
    pub qwen35: bool,
    #[serde(skip)]
    pub prefix: String,
    #[serde(skip)]
    pub rope_theta: f32,
    #[serde(skip)]
    pub rotary_dim: usize,
}

impl TextConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &serde_json::Value) -> Result<Self> {
        let kind = v["model_type"].as_str().unwrap_or("");
        ensure!(
            matches!(kind, "qwen3" | "qwen3_5" | "qwen3_5_text"),
            "unsupported model type {kind}"
        );
        let nested = kind == "qwen3_5";
        let text = if nested { &v["text_config"] } else { &v };
        let mut c: Self = serde_json::from_value(text.clone())?;
        c.qwen35 = kind != "qwen3";
        c.prefix = if nested {
            "model.language_model"
        } else {
            "model"
        }
        .into();
        if let Some(tie) = v["tie_word_embeddings"].as_bool() {
            c.tie_word_embeddings = tie;
        }
        let rope = if text["rope_parameters"].is_object() {
            &text["rope_parameters"]
        } else {
            text
        };
        let rope_type = rope["rope_type"].as_str().unwrap_or("default");
        ensure!(
            rope_type == "default",
            "unsupported RoPE scaling {rope_type}"
        );
        ensure!(
            text["rope_scaling"].is_null(),
            "Qwen3 scaled RoPE is not implemented"
        );
        c.rope_theta =
            rope["rope_theta"]
                .as_f64()
                .unwrap_or(if c.qwen35 { 10_000_000. } else { 1_000_000. }) as f32;
        let factor = rope["partial_rotary_factor"].as_f64().unwrap_or(1.);
        c.rotary_dim = (c.head_dim as f64 * factor) as usize;
        ensure!(
            c.hidden_size > 0
                && c.intermediate_size > 0
                && c.num_hidden_layers > 0
                && c.vocab_size > 0,
            "invalid model dimensions"
        );
        ensure!(
            c.num_key_value_heads > 0
                && c.num_attention_heads > 0
                && c.num_attention_heads % c.num_key_value_heads == 0,
            "invalid GQA heads"
        );
        ensure!(
            c.head_dim > 0
                && c.head_dim <= 256
                && c.rotary_dim > 0
                && c.rotary_dim <= c.head_dim
                && c.rotary_dim % 2 == 0,
            "unsupported attention dimensions"
        );
        ensure!(
            c.rope_theta.is_finite()
                && c.rope_theta > 0.
                && c.rms_norm_eps.is_finite()
                && c.rms_norm_eps > 0.,
            "invalid numerical configuration"
        );
        ensure!(
            text["use_sliding_window"].as_bool() != Some(true)
                && text["attention_bias"].as_bool() != Some(true),
            "sliding attention/bias unsupported"
        );
        if c.layer_types.is_empty() {
            c.layer_types = vec!["full_attention".into(); c.num_hidden_layers];
        }
        ensure!(
            c.layer_types.len() == c.num_hidden_layers,
            "layer_types length mismatch"
        );
        for kind in &c.layer_types {
            ensure!(
                kind == "full_attention" || (c.qwen35 && kind == "linear_attention"),
                "unsupported layer type {kind}"
            );
        }
        if c.qwen35 {
            ensure!(
                text["attn_output_gate"].as_bool() != Some(false),
                "ungated Qwen3.5 attention unsupported"
            );
            ensure!(
                c.linear_num_key_heads > 0
                    && c.linear_num_value_heads % c.linear_num_key_heads == 0
                    && c.linear_num_value_heads > 0,
                "invalid DeltaNet heads"
            );
            ensure!(
                (1..=256).contains(&c.linear_key_head_dim)
                    && c.linear_value_head_dim > 0
                    && c.linear_conv_kernel_dim > 0,
                "invalid DeltaNet dimensions"
            );
        }
        Ok(c)
    }

    pub fn shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut shapes = vec![
            (
                format!("{}.embed_tokens.weight", self.prefix),
                vec![self.vocab_size, self.hidden_size],
            ),
            (
                format!("{}.norm.weight", self.prefix),
                vec![self.hidden_size],
            ),
        ];
        if !self.tie_word_embeddings {
            shapes.push((
                "lm_head.weight".into(),
                vec![self.vocab_size, self.hidden_size],
            ));
        }
        for (i, kind) in self.layer_types.iter().enumerate() {
            let p = format!("{}.layers.{i}", self.prefix);
            for name in ["input_layernorm", "post_attention_layernorm"] {
                shapes.push((format!("{p}.{name}.weight"), vec![self.hidden_size]));
            }
            for (name, shape) in [
                ("gate_proj", vec![self.intermediate_size, self.hidden_size]),
                ("up_proj", vec![self.intermediate_size, self.hidden_size]),
                ("down_proj", vec![self.hidden_size, self.intermediate_size]),
            ] {
                shapes.push((format!("{p}.mlp.{name}.weight"), shape));
            }
            if kind == "full_attention" {
                for (name, shape) in [
                    (
                        "q_proj",
                        vec![
                            self.num_attention_heads
                                * self.head_dim
                                * if self.qwen35 { 2 } else { 1 },
                            self.hidden_size,
                        ],
                    ),
                    (
                        "k_proj",
                        vec![self.num_key_value_heads * self.head_dim, self.hidden_size],
                    ),
                    (
                        "v_proj",
                        vec![self.num_key_value_heads * self.head_dim, self.hidden_size],
                    ),
                    (
                        "o_proj",
                        vec![self.hidden_size, self.num_attention_heads * self.head_dim],
                    ),
                    ("q_norm", vec![self.head_dim]),
                    ("k_norm", vec![self.head_dim]),
                ] {
                    shapes.push((format!("{p}.self_attn.{name}.weight"), shape));
                }
            } else {
                let kd = self.linear_num_key_heads * self.linear_key_head_dim;
                let vd = self.linear_num_value_heads * self.linear_value_head_dim;
                for (name, shape) in [
                    ("in_proj_qkv.weight", vec![2 * kd + vd, self.hidden_size]),
                    ("in_proj_z.weight", vec![vd, self.hidden_size]),
                    (
                        "in_proj_a.weight",
                        vec![self.linear_num_value_heads, self.hidden_size],
                    ),
                    (
                        "in_proj_b.weight",
                        vec![self.linear_num_value_heads, self.hidden_size],
                    ),
                    ("out_proj.weight", vec![self.hidden_size, vd]),
                    (
                        "conv1d.weight",
                        vec![2 * kd + vd, 1, self.linear_conv_kernel_dim],
                    ),
                    ("A_log", vec![self.linear_num_value_heads]),
                    ("dt_bias", vec![self.linear_num_value_heads]),
                    ("norm.weight", vec![self.linear_value_head_dim]),
                ] {
                    shapes.push((format!("{p}.linear_attn.{name}"), shape));
                }
            }
        }
        shapes
    }
}
