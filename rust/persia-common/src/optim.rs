use std::sync::Arc;

use persia_libs::{hashbrown::HashMap, ndarray, parking_lot::RwLock};

use persia_embedding_config::EmbeddingConfig;
use persia_simd::{
    adam_avx2, decayed_adagrad_avx2, decayed_adagrad_vectorwise_shared_avx2, decayed_sgd_avx2,
};
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
    pub vectorwise_shared: bool,
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
        emb_opt_state: &Option<Vec<f32>>,
    );

    fn get_emb_state(&self, state: &Option<Vec<f32>>, idx: usize) -> Option<Vec<f32>> {
        None
    }

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
                    .slots_config
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
    fn get_emb_state(&self, opt_state: &Option<Vec<f32>>, idx: usize) -> Option<Vec<f32>> {
        if let Some(opt_state) = opt_state {
            Some(vec![opt_state[idx], opt_state[opt_state.len() / 2 + idx]])
        } else {
            None
        }
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
        emb_opt_state: &Option<Vec<f32>>,
    ) {
        let emb_opt_state = emb_opt_state.as_deref().unwrap();
        let beta1_power = emb_opt_state[0];
        let beta2_power = emb_opt_state[1];
        let (emb, opt) = emb_entry.split_at_mut(dim);
        let (adam_m, adam_v) = opt.split_at_mut(dim);

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
        match self.config.vectorwise_shared {
            true => 1,
            false => dim,
        }
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
        if self.config.vectorwise_shared {
            let adagrad_state = adagrad_state.first_mut().expect("adagrad state is empty");
            unsafe {
                decayed_adagrad_vectorwise_shared_avx2(
                    *adagrad_state,
                    emb,
                    grad,
                    self.config.lr,
                    self.config.eps,
                )
            }

            let gradients = ndarray::ArrayView1::<f32>::from(grad);

            let gradient_squares = gradients.dot(&gradients) / gradients.len() as f32;
            *adagrad_state = *adagrad_state * self.config.g_square_momentum + gradient_squares;
        } else {
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

#[cfg(test)]
mod sparse_optimizer_tests {
    // importing names from outer (for mod tests) scope.
    use super::*;

    fn get_grads() -> Vec<Vec<f32>> {
        vec![
            vec![
                0.6039, 0.2480, 0.8303, 0.8006, 0.6830, 0.4730, 0.0381, 0.8375, 0.5836, 0.8673,
                0.2224, 0.4040,
            ],
            vec![
                0.4478, 0.9670, 0.5724, 0.3074, 0.5760, 0.2937, 0.0995, 0.6640, 0.7718, 0.3016,
                0.0246, 0.6975,
            ],
            vec![
                0.2304, 0.9627, 0.3126, 0.8667, 0.6767, 0.6441, 0.0131, 0.1702, 0.8901, 0.4696,
                0.2655, 0.0545,
            ],
        ]
    }

    fn get_init_embedding() -> Vec<f32> {
        vec![
            0.7306, 0.0340, 0.1331, 0.4355, 0.0305, 0.6968, 0.1528, 0.7074, 0.5598, 0.0271, 0.7671,
            0.8731,
        ]
    }

    fn get_embedding_dim() -> usize {
        12
    }

    fn execute_test(optimizer: Box<dyn Optimizable>) -> Vec<f32> {
        let embedding_dim = get_embedding_dim();
        let mut embedding_entry = get_init_embedding();

        embedding_entry.resize(
            embedding_dim + optimizer.require_space(embedding_dim),
            0.0_f32,
        );

        optimizer.state_initialization(&mut embedding_entry, embedding_dim);

        let grads = get_grads();

        grads.iter().for_each(|g| {
            optimizer.update(&mut embedding_entry, g.as_slice(), embedding_dim, &None);
        });

        embedding_entry
    }

    #[test]
    fn test_adagrad() {
        let optimizer = Adagrad {
            config: AdagradConfig {
                lr: 0.01_f32,
                wd: 0.0_f32,
                g_square_momentum: 1.0_f32,
                initialization: 0.01_f32,
                eps: 1e-10_f32,
                vectorwise_shared: false,
            },
        };

        let embedding_entry = execute_test(Box::new(optimizer));

        let adagrad_result: Vec<f32> = vec![
            0.6598564,
            -0.036559787,
            0.04014046,
            0.34159237,
            -0.053671654,
            0.6320387,
            0.1387946,
            0.6141905,
            0.47925496,
            -0.06816861,
            0.7330182,
            0.81526995,
            0.6283042,
            1.9333843,
            1.1247585,
            1.496624,
            1.2661879,
            0.7348535,
            0.021523468,
            1.1812702,
            1.7385421,
            1.073696,
            0.13055718,
            0.6626925,
        ];

        embedding_entry
            .iter()
            .zip(adagrad_result.iter())
            .for_each(|(x, y)| assert_eq!(x, y));
    }

    #[test]
    fn test_adagrad_vectorwise_shared() {
        let optimizer = Adagrad {
            config: AdagradConfig {
                lr: 0.01_f32,
                wd: 0.0_f32,
                g_square_momentum: 1.0_f32,
                initialization: 0.01_f32,
                eps: 1e-10_f32,
                vectorwise_shared: true,
            },
        };

        let embedding_entry = execute_test(Box::new(optimizer));

        let adagrad_result: Vec<f32> = vec![
            0.6601662,
            -0.018124206,
            0.03701234,
            0.33996183,
            -0.055326782,
            0.63694036,
            0.14721976,
            0.6108338,
            0.47815663,
            -0.070203856,
            0.741245,
            0.82074344,
            0.99936616,
        ];

        embedding_entry
            .iter()
            .zip(adagrad_result.iter())
            .for_each(|(x, y)| assert_eq!(x, y));
    }
}
