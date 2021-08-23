use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

use persia_embedding_config::EmbeddingConfig;
use persia_simd::{adam_avx2, decayed_adagrad_avx2, decayed_sgd_avx2};

use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, Debug, Clone)]
pub enum OptimizerConfig {
    Adam(AdamConfig),
    SGD(NaiveSGDConfig),
    Adagrad(AdagradConfig),
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct NaiveSGDConfig {
    pub lr: f32,
    pub wd: f32,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct AdagradConfig {
    pub lr: f32,
    pub wd: f32,
    pub g_square_momentum: f32,
    pub initialization: f32,
    pub eps: f32,
}

pub enum Optimizer {
    Adam(Adam),
    SGD(NaiveSGD),
    Adagrad(Adagrad),
}

impl Optimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        match config {
            OptimizerConfig::SGD(config) => Self::SGD(NaiveSGD { config }),
            OptimizerConfig::Adagrad(config) => Self::Adagrad(Adagrad { config }),
            OptimizerConfig::Adam(config) => Self::Adam(Adam::new(config)),
        }
    }

    pub fn to_optimizable(self) -> Box<dyn Optimizable + Send + Sync> {
        match self {
            Optimizer::Adagrad(val) => Box::new(val) as Box<dyn Optimizable + Send + Sync>,
            Optimizer::SGD(val) => Box::new(val) as Box<dyn Optimizable + Send + Sync>,
            Optimizer::Adam(val) => Box::new(val) as Box<dyn Optimizable + Send + Sync>,
        }
    }
}

pub trait Optimizable {
    fn update(
        &self,
        emb_entry: &mut [f32],
        grad: &[f32],
        dim: usize,
        batch_level_status: &Option<Vec<f32>>,
    );

    fn get_batch_level_state(&self, _signs: &[u64]) -> Option<Vec<f32>> {
        None
    }

    #[inline]
    fn require_space(&self, _dim: usize) -> usize {
        0
    }

    // fn update_step(&mut self, feature_space: &str) {}
    fn update_lr(&mut self, _lr: f32) {}

    fn state_initialization(&self, _state: &mut [f32], _dim: usize) {}
}

struct AdamPowerOfBetas {
    beta1: f32,
    beta2: f32,
}

pub struct Adam {
    config: AdamConfig,
    embedding_config: Arc<EmbeddingConfig>,
    accum_betas: HashMap<u64, RwLock<AdamPowerOfBetas>>,
}

impl Adam {
    pub fn new(config: AdamConfig) -> Self {
        let embedding_config = EmbeddingConfig::get().expect("embedding config not found");
        let mut accum_betas = HashMap::with_capacity(embedding_config.feature_groups.len());

        embedding_config
            .feature_groups
            .iter()
            .for_each(|(_group_name, slots_name)| {
                let prefix = embedding_config
                    .slot_configs
                    .get(slots_name.first().expect("slot not found"))
                    .expect("slot not found")
                    .index_prefix;

                let initially_betas = AdamPowerOfBetas {
                    beta1: config.beta1.clone(),
                    beta2: config.beta2.clone(),
                };
                accum_betas.insert(prefix.clone(), RwLock::new(initially_betas));
            });

        Self {
            config,
            embedding_config,
            accum_betas,
        }
    }
}

impl Optimizable for Adam {
    #[inline]
    fn require_space(&self, dim: usize) -> usize {
        dim * 2
    }

    #[inline]
    fn get_batch_level_state(&self, signs: &[u64]) -> Option<Vec<f32>> {
        let mut betas_power = vec![0.0_f32; signs.len() * 2];
        let (beta1_power, beta2_power) = betas_power.as_mut_slice().split_at_mut(signs.len());

        let mask =
            !((1u64 << (u64::BITS - self.embedding_config.feature_index_prefix_bit as u32)) - 1);
        let mut steped: HashMap<u64, AdamPowerOfBetas> = HashMap::with_capacity(signs.len());

        signs.iter().enumerate().for_each(|(idx, sign)| {
            let masked_sign: u64 = sign & mask;
            if let Some(betas) = steped.get(&masked_sign) {
                beta1_power[idx] = betas.beta1;
                beta2_power[idx] = betas.beta2;
            } else {
                {
                    let mut accum_betas = self
                        .accum_betas
                        .get(&masked_sign)
                        .expect("feature group not found")
                        .write();

                    accum_betas.beta1 = accum_betas.beta1 * self.config.beta1;
                    accum_betas.beta2 = accum_betas.beta2 * self.config.beta2;

                    beta1_power[idx] = accum_betas.beta1;
                    beta2_power[idx] = accum_betas.beta2;
                }

                steped.insert(
                    masked_sign,
                    AdamPowerOfBetas {
                        beta1: beta1_power[idx],
                        beta2: beta2_power[idx],
                    },
                );
            }
        });

        Some(betas_power)
    }

    #[inline]
    fn update(
        &self,
        emb_entry: &mut [f32],
        grad: &[f32],
        dim: usize,
        batch_level_status: &Option<Vec<f32>>,
    ) {
        let batch_level_status = batch_level_status.as_deref().unwrap();
        let (emb, opt) = emb_entry.split_at_mut(dim);
        let (adam_m, adam_v) = opt.split_at_mut(dim);
        let (beta1_power, beta2_power) = batch_level_status.split_at(dim);

        unsafe {
            adam_avx2(
                adam_m,
                adam_v,
                beta1_power,
                beta2_power,
                emb,
                grad,
                self.config.lr,
                self.config.beta1,
                self.config.beta2,
                self.config.eps,
            )
        }
    }
}

pub struct NaiveSGD {
    config: NaiveSGDConfig,
}

impl Optimizable for NaiveSGD {
    #[inline]
    fn update(
        &self,
        emb_entry: &mut [f32],
        grad: &[f32],
        _dim: usize,
        _batch_level_status: &Option<Vec<f32>>,
    ) {
        unsafe {
            decayed_sgd_avx2(emb_entry, grad, self.config.wd, self.config.lr);
        }
    }

    fn update_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

pub struct Adagrad {
    config: AdagradConfig,
}

impl Optimizable for Adagrad {
    #[inline]
    fn require_space(&self, dim: usize) -> usize {
        dim
    }

    #[inline]
    fn update(
        &self,
        emb_entry: &mut [f32],
        grad: &[f32],
        dim: usize,
        _batch_level_status: &Option<Vec<f32>>,
    ) {
        let (emb, adagrad_state) = emb_entry.split_at_mut(dim);
        unsafe {
            decayed_adagrad_avx2(
                adagrad_state,
                emb,
                grad,
                self.config.g_square_momentum,
                self.config.lr,
                self.config.eps,
            )
        }
    }

    #[inline]
    fn state_initialization(&self, state: &mut [f32], dim: usize) {
        state[dim..]
            .copy_from_slice(vec![self.config.initialization; self.require_space(dim)].as_slice());
    }

    fn update_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}
