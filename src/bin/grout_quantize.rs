use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// Produce a separate packed Q4 checkpoint without modifying the source.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 64)]
    group_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    grout::quantization::convert(&args.model, &args.output, args.group_size)
}
