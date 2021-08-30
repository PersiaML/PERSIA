use persia_libs::{half, ndarray::Array2};
use serde::{Deserialize, Serialize};

use persia_speedy::{Readable, Writable};

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub enum Gradients {
    F16(Array2<half::f16>),
    F32(Array2<f32>),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct FeatureEmbeddingGradientBatch {
    pub feature_name: String,
    pub gradients: Gradients,
    pub scale_factor: f32,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct SkippedGradientBatch {
    pub feature_name: String,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub enum SkippableFeatureEmbeddingGradientBatch {
    GradientBatch(FeatureEmbeddingGradientBatch),
    Skipped(SkippedGradientBatch),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct EmbeddingGradientBatch {
    pub gradients: Vec<SkippableFeatureEmbeddingGradientBatch>,
}
