use anyhow::Result;
use clap::Parser;
use grout::engine::Engine;
use grout::mtp::{MtpOptions, Strategy};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Native Qwen3 and quantized Qwen3.5 text inference"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    max_seq_len: Option<usize>,

    #[arg(long, default_value_t = false)]
    sample: bool,

    #[arg(long, default_value_t = false)]
    raw_prompt: bool,

    /// Enable thinking in the packed backend's single-user chat template.
    #[arg(long, default_value_t = false)]
    enable_thinking: bool,

    #[arg(long, default_value_t = false)]
    device_argmax: bool,

    #[arg(long, default_value_t = false)]
    profile: bool,

    /// Discarded warmup generations before the measured run. The first
    /// generate() pays JIT compile + decode-graph capture (~0.85s of cold
    /// prefill), which otherwise lands in the reported prompt t/s. The
    /// default 1 warmup makes the reported t/s reflect steady state; set 0
    /// to see cold-start numbers.
    #[arg(long, default_value_t = 1)]
    warmup_reps: usize,

    /// Enable greedy multi-token prediction.
    #[arg(long, conflicts_with = "sample")]
    do_mtp: bool,
    #[arg(long, default_value_t = 16, requires = "do_mtp")]
    k_toks: usize,
    #[arg(long, requires = "do_mtp")]
    mask_id: Option<u32>,
    #[arg(long, default_value = "static", value_parser = ["static", "conf_adapt"], requires = "do_mtp")]
    strategy: String,
    #[arg(long, default_value_t = 0.9, requires = "do_mtp")]
    confidence_threshold: f32,
    /// Save token IDs and MTP chunk lengths for reference comparisons.
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut engine = Engine::load(&args.model, args.max_seq_len).await?;
    engine.set_sampling_enabled(args.sample);
    engine.set_chat_template_enabled(!args.raw_prompt);
    engine.set_device_argmax_enabled(args.device_argmax);
    engine.set_profile_enabled(args.profile);
    engine.set_thinking_enabled(args.enable_thinking)?;
    let mtp = if args.do_mtp {
        Some(MtpOptions {
            k: args.k_toks,
            mask_id: args
                .mask_id
                .map(Ok)
                .unwrap_or_else(|| engine.mtp_mask_id())?,
            strategy: if args.strategy == "conf_adapt" {
                Strategy::ConfAdapt {
                    threshold: args.confidence_threshold,
                }
            } else {
                Strategy::Static
            },
        })
    } else {
        None
    };

    println!("Loaded model from {}", args.model.display());
    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...", args.max_new_tokens);

    // Warmup (discarded): the first generate() pays JIT compile + decode-graph
    // capture, which would otherwise pollute the reported prompt t/s. A few
    // decode tokens are enough to JIT the prefill path and capture/replay the
    // decode graph, so cap warmup length to keep it cheap. --warmup-reps 0 opts out.
    let warmup_tokens = args.max_new_tokens.min(8);
    for _ in 0..args.warmup_reps {
        if let Some(options) = mtp {
            let _ = engine
                .generate_mtp(&args.prompt, warmup_tokens, options)
                .await?;
        } else {
            let _ = engine.generate(&args.prompt, warmup_tokens).await?;
        }
    }

    let output = if let Some(options) = mtp {
        engine
            .generate_mtp(&args.prompt, args.max_new_tokens, options)
            .await?
    } else {
        engine.generate(&args.prompt, args.max_new_tokens).await?
    };
    if !output.mtp_chunks.is_empty() {
        println!(
            "MTP: {} passes, mean {:.2} tokens/pass, chunks {:?}",
            output.mtp_chunks.len(),
            output.generated_tokens as f64 / output.mtp_chunks.len() as f64,
            output.mtp_chunks
        );
    }
    println!();
    println!("{}", output.text);
    println!(
        "t/s: {:.2} prompt, {:.2} decode phase, {:.2} end-to-end (prompt_tokens={}, generated_tokens={}, prompt_s={:.3}, decode_s={:.3}, total_s={:.3})",
        output.prompt_tps(),
        output.decode_phase_tps(),
        output.total_tps(),
        output.prompt_tokens,
        output.generated_tokens,
        output.prompt_elapsed.as_secs_f64(),
        output.decode_elapsed.as_secs_f64(),
        output.total_elapsed.as_secs_f64(),
    );
    if let Some(report) = output.profile_report {
        println!();
        println!("{report}");
    }
    if let Some(path) = args.output_json {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "text":output.text,"token_ids":output.token_ids,"mtp_chunks":output.mtp_chunks,
                "prompt_tokens":output.prompt_tokens,"generated_tokens":output.generated_tokens,
                "total_seconds":output.total_elapsed.as_secs_f64()
            }))?,
        )?;
    }
    Ok(())
}
