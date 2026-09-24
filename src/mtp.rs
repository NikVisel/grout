//! Shared, backend-independent MTP scheduling and training layouts.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub enum Strategy {
    Static,
    ConfAdapt { threshold: f32 },
}

#[derive(Clone, Copy, Debug)]
pub struct MtpOptions {
    pub k: usize,
    pub mask_id: u32,
    pub strategy: Strategy,
}

impl MtpOptions {
    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        ensure!(self.k > 0, "MTP k must be positive");
        ensure!(
            (self.mask_id as usize) < vocab_size,
            "mask ID is outside vocabulary"
        );
        if let Strategy::ConfAdapt { threshold } = self.strategy {
            ensure!(
                threshold.is_finite() && (0.0..=1.0).contains(&threshold),
                "confidence threshold must be in [0, 1]"
            );
        }
        Ok(())
    }
}

/// Unscaled softmax confidence; ties select the lowest vocabulary index.
pub fn top1(logits: &[f32]) -> Result<(u32, f32)> {
    ensure!(!logits.is_empty(), "empty logits");
    ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    let z: f32 = logits.iter().map(|x| (x - logits[best]).exp()).sum();
    Ok((best as u32, 1.0 / z))
}

pub fn accepted_count(confidences: &[f32], strategy: Strategy) -> Result<usize> {
    ensure!(!confidences.is_empty(), "empty candidate block");
    ensure!(
        confidences
            .iter()
            .all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
        "invalid confidence"
    );
    Ok(match strategy {
        Strategy::Static => confidences.len(),
        Strategy::ConfAdapt { threshold } => {
            ensure!(
                threshold.is_finite() && (0.0..=1.0).contains(&threshold),
                "invalid threshold"
            );
            // Reference rejects < threshold, and always emits the first token.
            confidences
                .iter()
                .position(|&c| c < threshold)
                .unwrap_or(confidences.len())
                .max(1)
        }
    })
}

/// A packed training example. Ground-truth positions keep their original
/// RoPE indices; inserted positions belong to one isolated causal region.
#[derive(Clone, Debug)]
pub struct MtpBatch {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub regions: Vec<Option<usize>>,
    pub prediction_rows: Vec<usize>,
    pub mask_rows: Vec<usize>,
    pub k: usize,
}

impl MtpBatch {
    pub fn new(tokens: &[u32], anchors: &[usize], k: usize, mask_id: u32) -> Result<Self> {
        ensure!(
            k > 0 && !tokens.is_empty() && !anchors.is_empty(),
            "empty MTP layout or zero k"
        );
        ensure!(
            anchors.windows(2).all(|a| a[0] < a[1]),
            "anchors must be strictly increasing"
        );
        ensure!(
            anchors.iter().all(|&a| a < tokens.len()),
            "anchor outside input"
        );
        let mut b = Self {
            tokens: vec![],
            positions: vec![],
            regions: vec![],
            prediction_rows: vec![],
            mask_rows: vec![],
            k,
        };
        let mut region = 0;
        for (pos, &token) in tokens.iter().enumerate() {
            let row = b.tokens.len();
            b.tokens.push(token);
            b.positions.push(pos as u32);
            b.regions.push(None);
            if anchors.get(region) == Some(&pos) {
                b.prediction_rows.push(row);
                for j in 1..k {
                    b.prediction_rows.push(b.tokens.len());
                    b.mask_rows.push(b.tokens.len());
                    b.tokens.push(mask_id);
                    b.positions.push((pos + j) as u32);
                    b.regions.push(Some(region));
                }
                region += 1;
            }
        }
        Ok(b)
    }

    pub fn visible(&self, query: usize, key: usize) -> bool {
        key <= query
            && (self.regions[key].is_none()
                || (self.regions[query].is_some() && self.regions[query] == self.regions[key]))
    }

    pub fn attention_bias(&self) -> Vec<f32> {
        (0..self.tokens.len())
            .flat_map(|q| {
                (0..self.tokens.len()).map(move |key| {
                    if self.visible(q, key) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect()
    }

    /// Teacher conditions on student predictions, never ground-truth suffixes
    /// inside a prediction region. The final prediction need not be fed back.
    pub fn teacher_tokens(&self, proposals: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            proposals.len() == self.prediction_rows.len(),
            "proposal count differs from prediction rows"
        );
        let mut tokens = self.tokens.clone();
        if self.k > 1 {
            for (region, rows) in self.mask_rows.chunks(self.k - 1).enumerate() {
                for (j, &row) in rows.iter().enumerate() {
                    tokens[row] = proposals[region * self.k + j];
                }
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_prefix_and_fallback_match_reference() -> Result<()> {
        let s = Strategy::ConfAdapt { threshold: 0.9 };
        assert_eq!(accepted_count(&[0.95, 0.9, 0.85, 0.99], s)?, 2);
        assert_eq!(accepted_count(&[0.8, 0.99, 0.99], s)?, 1);
        assert_eq!(accepted_count(&[0.95; 4], s)?, 4);
        assert_eq!(accepted_count(&[0.1; 4], Strategy::Static)?, 4);
        assert!(accepted_count(&[f32::NAN], s).is_err());
        Ok(())
    }
    #[test]
    fn confidence_is_stable_and_ties_are_first() -> Result<()> {
        let (id, p) = top1(&[10000., 10000.])?;
        assert_eq!(id, 0);
        assert!((p - 0.5).abs() < 1e-6);
        Ok(())
    }
    #[test]
    fn packed_regions_preserve_positions_and_isolate_rollouts() -> Result<()> {
        let b = MtpBatch::new(&[10, 11, 12, 13, 14, 15], &[1, 4], 3, 99)?;
        assert_eq!(b.tokens, [10, 11, 99, 99, 12, 13, 14, 99, 99, 15]);
        assert_eq!(b.positions, [0, 1, 2, 3, 2, 3, 4, 5, 6, 5]);
        assert_eq!(b.prediction_rows, [1, 2, 3, 6, 7, 8]);
        assert!(!b.visible(4, 2));
        assert!(!b.visible(8, 3));
        assert!(b.visible(8, 7));
        assert!(b.visible(8, 5));
        assert!(!b.visible(7, 8));
        assert!(b.visible(3, 2));
        assert_eq!(
            b.teacher_tokens(&[20, 21, 22, 23, 24, 25])?,
            [10, 11, 20, 21, 12, 13, 14, 23, 24, 15]
        );
        Ok(())
    }
}
