//! Portable affine Q4 checkpoint format. Two unsigned codes per byte, low nibble
//! first; each row/group has independent FP32 (scale, offset). GPU weights stay
//! packed. This is not NF4, FP4, GPTQ, AWQ, or a standard HF quantization format.
use anyhow::{Context, Result, bail, ensure};
use cutile::core::{bf16, f16};
use memmap2::MmapOptions;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

pub const FORMAT: &str = "grout-affine-q4-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub file: String,
    pub shape: Vec<usize>,
    pub quantized: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub group_size: usize,
    pub text_only: bool,
    pub tensors: BTreeMap<String, TensorSpec>,
    pub source_bytes: u64,
    pub stored_bytes: u64,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self> {
        let m: Self = serde_json::from_slice(&fs::read(dir.join("quantization.json"))?)?;
        ensure!(
            m.format == FORMAT,
            "unsupported quantization format {}",
            m.format
        );
        ensure!(
            m.group_size >= 2 && m.group_size % 2 == 0,
            "invalid Q4 group size"
        );
        for spec in m.tensors.values() {
            ensure!(
                Path::new(&spec.file).components().count() == 1
                    && !Path::new(&spec.file).is_absolute(),
                "invalid tensor file path"
            );
            ensure!(
                !spec.shape.is_empty() && spec.shape.iter().all(|&n| n > 0),
                "invalid tensor shape"
            );
        }
        Ok(m)
    }
}

pub fn decode(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        _ => bail!("unsupported source dtype {dtype:?}"),
    })
}

pub struct PackedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub codes: Vec<u8>,
    /// Interleaved scale and offset for each row/group.
    pub scales: Vec<f32>,
}

impl PackedMatrix {
    pub fn quantize(values: &[f32], rows: usize, cols: usize, group_size: usize) -> Result<Self> {
        ensure!(
            rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(values.len()),
            "invalid matrix shape"
        );
        ensure!(
            group_size >= 2 && group_size % 2 == 0,
            "group size must be positive and even"
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "cannot quantize non-finite weights"
        );
        let groups = cols.div_ceil(group_size);
        let stride = cols.div_ceil(2);
        let mut codes = vec![0u8; rows * stride];
        let mut scales = vec![0f32; rows * groups * 2];
        codes
            .par_chunks_mut(stride)
            .zip(scales.par_chunks_mut(groups * 2))
            .zip(values.par_chunks(cols))
            .for_each(|((dst, params), src)| {
                for (group, xs) in src.chunks(group_size).enumerate() {
                    let lo = xs.iter().copied().fold(f32::INFINITY, f32::min);
                    let hi = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = (hi - lo) / 15.0;
                    params[group * 2] = scale;
                    params[group * 2 + 1] = lo;
                    for (j, &x) in xs.iter().enumerate() {
                        let q = if scale == 0.0 {
                            0
                        } else {
                            ((x - lo) / scale).round().clamp(0.0, 15.0) as u8
                        };
                        let col = group * group_size + j;
                        dst[col / 2] |= q << ((col % 2) * 4);
                    }
                }
            });
        Ok(Self {
            rows,
            cols,
            group_size,
            codes,
            scales,
        })
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        let q = (self.codes[row * self.cols.div_ceil(2) + col / 2] >> ((col % 2) * 4)) & 15;
        let g = (row * self.cols.div_ceil(self.group_size) + col / self.group_size) * 2;
        q as f32 * self.scales[g] + self.scales[g + 1]
    }
}

fn float_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One source tensor at a time; no model-sized FP32 copy or GPU allocation.
pub fn convert(source: &Path, output: &Path, group_size: usize) -> Result<()> {
    ensure!(
        group_size >= 2 && group_size % 2 == 0,
        "group size must be positive and even"
    );
    ensure!(
        !output.exists(),
        "output already exists; choose a new directory"
    );
    let config: serde_json::Value = serde_json::from_slice(&fs::read(source.join("config.json"))?)?;
    let kind = config["model_type"].as_str().unwrap_or("");
    ensure!(
        matches!(kind, "qwen3" | "qwen3_5" | "qwen3_5_text"),
        "unsupported model type {kind}"
    );
    ensure!(
        !source.join("quantization.json").exists(),
        "source is already quantized"
    );
    let index: crate::config::SafetensorsIndex =
        serde_json::from_slice(&fs::read(source.join("model.safetensors.index.json"))?)?;
    for name in ["tokenizer.json", "config.json"] {
        ensure!(source.join(name).is_file(), "missing {name}");
    }
    fs::create_dir_all(output)?;
    let mut manifest = Manifest {
        format: FORMAT.into(),
        group_size,
        text_only: true,
        tensors: BTreeMap::new(),
        source_bytes: 0,
        stored_bytes: 0,
    };
    let mut weight_map = BTreeMap::new();
    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, file) in index.weight_map {
        if name.starts_with("model.visual.") {
            continue;
        }
        by_shard.entry(file).or_default().push(name);
    }
    let mut number = 0;
    for (file, mut names) in by_shard {
        let source_file = fs::File::open(source.join(&file))?;
        let mmap = unsafe { MmapOptions::new().map(&source_file)? };
        let st = SafeTensors::deserialize(&mmap)?;
        names.sort();
        for name in names {
            let view = st.tensor(&name)?;
            let shape = view.shape().to_vec();
            let values = decode(view.dtype(), view.data()).with_context(|| name.clone())?;
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "non-finite tensor {name}"
            );
            let quantized = shape.len() == 2 && shape[1] >= group_size;
            let file = format!("weights-{number:05}.safetensors");
            let packed;
            let scales;
            let dense;
            let mut tensors = BTreeMap::new();
            if quantized {
                packed = PackedMatrix::quantize(&values, shape[0], shape[1], group_size)?;
                scales = float_bytes(&packed.scales);
                tensors.insert(
                    name.clone(),
                    TensorView::new(
                        Dtype::U8,
                        vec![shape[0], shape[1].div_ceil(2)],
                        &packed.codes,
                    )?,
                );
                let scale_name = format!("{name}.grout_scales");
                tensors.insert(
                    scale_name.clone(),
                    TensorView::new(
                        Dtype::F32,
                        vec![shape[0], shape[1].div_ceil(group_size), 2],
                        &scales,
                    )?,
                );
                weight_map.insert(scale_name, file.clone());
            } else {
                dense = float_bytes(&values);
                tensors.insert(
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), &dense)?,
                );
            }
            safetensors::serialize_to_file(tensors, None, &output.join(&file))?;
            manifest.source_bytes += view.data().len() as u64;
            manifest.stored_bytes += fs::metadata(output.join(&file))?.len();
            weight_map.insert(name.clone(), file.clone());
            manifest.tensors.insert(
                name.clone(),
                TensorSpec {
                    file,
                    shape,
                    quantized,
                },
            );
            number += 1;
            if number % 25 == 0 {
                println!(
                    "Converted {number} tensors ({:.2} GiB)",
                    manifest.stored_bytes as f64 / (1u64 << 30) as f64
                );
            }
        }
    }
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
        "mtp-source.json",
        "model-source.json",
        "training_run.json",
    ] {
        let path = source.join(name);
        if path.is_file() {
            fs::copy(path, output.join(name))?;
        }
    }
    fs::write(
        output.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"weight_map":weight_map}))?,
    )?;
    // Publish the manifest last; partial conversions cannot be loaded.
    fs::write(
        output.join("quantization.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "Saved {} tensors: {:.2} GiB -> {:.2} GiB",
        manifest.tensors.len(),
        manifest.source_bytes as f64 / (1u64 << 30) as f64,
        manifest.stored_bytes as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_handles_rows_groups_tail_and_constants() -> Result<()> {
        let values: Vec<f32> = (0..51)
            .map(|i| {
                if i >= 34 {
                    -3.25
                } else {
                    (i as f32 * 1.3).sin()
                }
            })
            .collect();
        let q = PackedMatrix::quantize(&values, 3, 17, 8)?;
        assert_eq!(q.codes.len(), 27);
        for r in 0..3 {
            for c in 0..17 {
                let g = (r * 3 + c / 8) * 2;
                assert!((q.get(r, c) - values[r * 17 + c]).abs() <= q.scales[g] * 0.501 + 1e-6);
            }
        }
        let q = PackedMatrix::quantize(&[0., 1., 14., 15.], 1, 4, 4)?;
        assert_eq!(q.codes, [0x10, 0xfe]);
        Ok(())
    }

    #[test]
    fn rejects_invalid_input() {
        assert!(PackedMatrix::quantize(&[f32::NAN], 1, 1, 2).is_err());
        assert!(PackedMatrix::quantize(&[1.], 1, 1, 3).is_err());
        assert!(PackedMatrix::quantize(&[1.], 2, 1, 2).is_err());
    }
}
