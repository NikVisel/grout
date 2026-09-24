use anyhow::{Context, Result, ensure};
use clap::Parser;
use grout::{
    mtp::MtpBatch,
    training::{
        NativeModel, NativeOptimizer, Supervision, TrainModel, ntp_loss, student_forced_loss,
    },
};
use rand::{Rng, SeedableRng};
use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
};

#[derive(Parser, Debug)]
#[command(about = "Native Rust Qwen3/Qwen3.5 student-forced MTP training (CPU f32)")]
struct Args {
    /// Initial student HF checkpoint, or an exported native checkpoint.
    #[arg(long, required_unless_present = "smoke_test")]
    student: Option<PathBuf>,
    /// Frozen teacher. Use the original Qwen3 checkpoint for paper reproduction.
    #[arg(long, required_unless_present = "smoke_test")]
    teacher: Option<PathBuf>,
    /// JSONL with either {"text":"..."} or {"token_ids":[...]} per document.
    #[arg(long, required_unless_present = "smoke_test")]
    data: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    /// Exercise full forward/backward/optimizer/export on a tiny deterministic Qwen3.
    #[arg(long)]
    smoke_test: bool,
    #[arg(long, default_value = "qwen3", value_parser = ["qwen3", "qwen3.5"], requires = "smoke_test")]
    smoke_architecture: String,
    #[arg(long, default_value_t = 10)]
    steps: usize,
    #[arg(long, default_value_t = 160)]
    sequence_length: usize,
    #[arg(long, default_value_t = 2)]
    k_min: usize,
    #[arg(long, default_value_t = 16)]
    k_max: usize,
    #[arg(long, default_value_t = 5)]
    regions: usize,
    #[arg(long)]
    mask_id: Option<u32>,
    #[arg(long, default_value = "hard", value_parser = ["hard", "soft", "ntp"])]
    objective: String,
    #[arg(long, default_value = "adamw", value_parser = ["adamw", "sgd"])]
    optimizer: String,
    #[arg(long, default_value_t = 1e-5)]
    learning_rate: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.steps > 0 && args.sequence_length >= 2 && args.regions > 0,
        "steps/regions must be positive; sequence length >= 2"
    );
    ensure!(
        args.k_min >= 1 && args.k_max >= args.k_min,
        "invalid k range"
    );
    ensure!(
        !args.output.exists(),
        "output exists: {}",
        args.output.display()
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let (student, teacher, documents, mask_id) = if args.smoke_test {
        (
            NativeModel::tiny(true, args.smoke_architecture == "qwen3.5")?,
            NativeModel::tiny(false, args.smoke_architecture == "qwen3.5")?,
            vec![vec![1, 2, 3, 4, 5, 6, 7, 8]],
            30,
        )
    } else {
        let path = args.student.as_ref().unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let teacher_path = args.teacher.as_ref().unwrap();
        let teacher_tokenizer =
            tokenizers::Tokenizer::from_file(teacher_path.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("teacher tokenizer: {e}"))?;
        // Teachers need not contain added mask tokens, but every shared token
        // must have the same ID. Proposals in added slots require compatible vocab.
        for (token, id) in teacher_tokenizer.get_vocab(true) {
            ensure!(
                tokenizer.token_to_id(&token) == Some(id),
                "teacher/student token ID mismatch for {token}"
            );
        }
        let mask_id = if args.objective == "ntp" {
            args.mask_id.unwrap_or(0)
        } else {
            args.mask_id
                .or_else(|| tokenizer.token_to_id("<|mtp_special_token_0|>"))
                .context("MTP mask token missing; use an extended checkpoint or --mask-id")?
        };
        let mut docs = Vec::new();
        for (line_number, line) in BufReader::new(std::fs::File::open(args.data.as_ref().unwrap())?)
            .lines()
            .enumerate()
        {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(&line)
                .with_context(|| format!("JSONL line {}", line_number + 1))?;
            let mut ids: Vec<u32> = if let Some(ids) = value.get("token_ids") {
                serde_json::from_value(ids.clone())?
            } else {
                tokenizer
                    .encode(
                        value["text"]
                            .as_str()
                            .context("document requires text or token_ids")?,
                        true,
                    )
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
                    .get_ids()
                    .to_vec()
            };
            // Preserve document boundaries and add the tokenizer's EOS. The
            // final MTP window can extend beyond EOS; later documents never leak in.
            if let Some(eos) = tokenizer.token_to_id("<|im_end|>") {
                ids.push(eos);
            }
            if ids.len() >= 2 {
                docs.push(ids);
            }
        }
        ensure!(!docs.is_empty(), "no nonempty documents");
        eprintln!(
            "Loading CPU f32 student and frozen teacher. Full AdamW state for 4B parameters exceeds 64 GB including gradients; use adequate RAM or --optimizer sgd."
        );
        (
            NativeModel::load(path, true)?,
            NativeModel::load(teacher_path, false)?,
            docs,
            mask_id,
        )
    };
    ensure!(
        (mask_id as usize) < student.vocab_size(),
        "mask outside vocabulary"
    );
    ensure!(
        student.compatible_with(&teacher),
        "teacher/student vocab sizes must match"
    );
    ensure!(
        documents
            .iter()
            .flatten()
            .all(|&t| (t as usize) < student.vocab_size()),
        "dataset token outside vocabulary"
    );
    let mut optimizer =
        NativeOptimizer::new(&student, args.optimizer == "adamw", args.learning_rate)?;
    for step in 0..args.steps {
        let doc = &documents[rng.gen_range(0..documents.len())];
        let start = if doc.len() > args.sequence_length {
            rng.gen_range(0..=doc.len() - args.sequence_length)
        } else {
            0
        };
        let tokens = &doc[start..(start + args.sequence_length).min(doc.len())];
        let k = rng.gen_range(args.k_min..=args.k_max);
        let loss = if args.objective == "ntp" {
            ntp_loss(&student, tokens)?
        } else {
            let stride = k.max(tokens.len() / args.regions).max(1);
            let offset = rng.gen_range(0..stride.min(tokens.len()));
            let anchors: Vec<_> = (offset..tokens.len())
                .step_by(stride)
                .take(args.regions)
                .collect();
            let batch = MtpBatch::new(tokens, &anchors, k, mask_id)?;
            ensure!(
                batch
                    .positions
                    .iter()
                    .all(|&p| (p as usize) < student.max_positions()),
                "training positions exceed context"
            );
            student_forced_loss(
                &student,
                &teacher,
                &batch,
                if args.objective == "soft" {
                    Supervision::Soft
                } else {
                    Supervision::Hard
                },
            )?
        };
        let value = loss.to_scalar::<f32>()?;
        optimizer.backward_step(&loss)?;
        println!("step={} loss={value:.6} k={k}", step + 1);
    }
    student.export(
        &args.output,
        if args.smoke_test {
            None
        } else {
            args.student.as_deref()
        },
    )?;
    std::fs::write(
        args.output.join("training_run.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "objective":args.objective,"optimizer":args.optimizer,"steps":args.steps,"seed":args.seed,
            "k_min":args.k_min,"k_max":args.k_max,"learning_rate":args.learning_rate,"mask_id":mask_id,
            "student":args.student,"teacher":args.teacher,"data":args.data,"smoke_test":args.smoke_test,
            "backend":"candle-cpu-f32","optimizer_state_saved":false
        }))?,
    )?;
    println!("Saved native checkpoint to {}", args.output.display());
    Ok(())
}
