use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub eos_token_id: u32,
}

impl Qwen3Config {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let cfg_text = fs::read_to_string(&cfg_path)
            .with_context(|| format!("failed to read {}", cfg_path.display()))?;
        let cfg: Self = serde_json::from_str(&cfg_text)
            .with_context(|| format!("failed to parse {}", cfg_path.display()))?;
        println!("cfg: {cfg:#?}");
        Ok(cfg)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[derive(Debug, Deserialize)]
pub struct SafetensorsIndex {
    pub weight_map: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(x) => vec![x],
            Self::Many(xs) => xs,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub eos_token_id: Option<EosTokenId>,
}

impl GenerationConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("generation_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(cfg))
    }
}
