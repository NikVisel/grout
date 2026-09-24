//! Select the packed backend from checkpoint metadata; keep the original FP16
//! Qwen3 engine and CUDA graph path available for unquantized checkpoints.
use crate::{
    model::{GenerationOutput, Qwen3Engine},
    mtp::MtpOptions,
    quantized::QuantizedEngine,
};
use anyhow::{Result, ensure};
use std::path::Path;

pub enum Engine {
    Qwen3(Qwen3Engine),
    Quantized(QuantizedEngine),
}
impl Engine {
    pub async fn load(dir: &Path, limit: Option<usize>) -> Result<Self> {
        if dir.join("quantization.json").exists() {
            return Ok(Self::Quantized(QuantizedEngine::load(dir, limit)?));
        }
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        ensure!(
            v["model_type"] == "qwen3",
            "Qwen3.5 requires a packed checkpoint: run grout_quantize --model <source> --output <new-directory>"
        );
        Ok(Self::Qwen3(Qwen3Engine::load(dir, limit).await?))
    }
    pub fn set_sampling_enabled(&mut self, v: bool) {
        match self {
            Self::Qwen3(e) => e.set_sampling_enabled(v),
            Self::Quantized(e) => e.set_sampling_enabled(v),
        }
    }
    pub fn set_chat_template_enabled(&mut self, v: bool) {
        match self {
            Self::Qwen3(e) => e.set_chat_template_enabled(v),
            Self::Quantized(e) => e.set_chat_template_enabled(v),
        }
    }
    pub fn set_device_argmax_enabled(&mut self, v: bool) {
        if let Self::Qwen3(e) = self {
            e.set_device_argmax_enabled(v)
        }
    }
    pub fn set_profile_enabled(&mut self, v: bool) {
        match self {
            Self::Qwen3(e) => e.set_profile_enabled(v),
            Self::Quantized(e) => e.set_profile_enabled(v),
        }
    }
    pub fn set_thinking_enabled(&mut self, v: bool) -> Result<()> {
        match self {
            Self::Quantized(e) => e.set_thinking_enabled(v),
            Self::Qwen3(_) => ensure!(
                !v,
                "use --raw-prompt to control thinking with the original FP16 backend"
            ),
        };
        Ok(())
    }
    pub fn mtp_mask_id(&self) -> Result<u32> {
        match self {
            Self::Qwen3(e) => e.mtp_mask_id(),
            Self::Quantized(e) => e.mtp_mask_id(),
        }
    }
    pub async fn generate(&mut self, prompt: &str, n: usize) -> Result<GenerationOutput> {
        match self {
            Self::Qwen3(e) => e.generate(prompt, n).await,
            Self::Quantized(e) => e.generate(prompt, n, None),
        }
    }
    pub async fn generate_mtp(
        &mut self,
        prompt: &str,
        n: usize,
        o: MtpOptions,
    ) -> Result<GenerationOutput> {
        match self {
            Self::Qwen3(e) => e.generate_mtp(prompt, n, o).await,
            Self::Quantized(e) => e.generate(prompt, n, Some(o)),
        }
    }
}
