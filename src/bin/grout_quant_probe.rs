//! Numerical validation interface: explicit tokens and prefill chunk boundaries.
use anyhow::{Result, ensure};
use clap::Parser;
use grout::quantized::QuantizedEngine;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<u32>,
    #[arg(long, value_delimiter = ',')]
    chunks: Vec<usize>,
    #[arg(long)]
    output: PathBuf,
    /// Exercise rollback by processing these temporary tokens after every chunk.
    #[arg(long, value_delimiter = ',')]
    temporary: Vec<u32>,
    #[arg(long)]
    last_only: bool,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(!a.tokens.is_empty(), "tokens required");
    let chunks = if a.chunks.is_empty() {
        vec![a.tokens.len()]
    } else {
        a.chunks
    };
    ensure!(
        chunks.iter().sum::<usize>() == a.tokens.len() && chunks.iter().all(|&n| n > 0 && n <= 256),
        "invalid chunks"
    );
    let mut engine =
        QuantizedEngine::load(&a.model, Some((a.tokens.len() + a.temporary.len()).max(32)))?;
    let mut logits = vec![];
    let mut offset = 0;
    for n in chunks {
        let rows = if a.last_only {
            if offset + n == a.tokens.len() { 1 } else { 0 }
        } else {
            n
        };
        logits.extend(engine.forward(&a.tokens[offset..offset + n], rows)?);
        offset += n;
        if !a.temporary.is_empty() {
            let snapshot = engine.snapshot()?;
            engine.forward(&a.temporary, 0)?;
            engine.restore(&snapshot)?;
        }
    }
    std::fs::write(
        a.output,
        serde_json::to_vec(
            &serde_json::json!({"logits":logits,"resident_weight_bytes":engine.resident_weight_bytes}),
        )?,
    )?;
    Ok(())
}
