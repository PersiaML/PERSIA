use persia_simd::{decayed_adagrad_avx2, decayed_sgd_avx2};

use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, Debug, Clone)]
pub enum Optimizer {
    Adam(Adam),
    SGD(NaiveSGD),
    Adagrad(Adagrad),
}

impl Optimizer {
    pub fn new_adagrad(
        lr: f32,
        wd: f32,
        g_square_momentum: f32,
        initialization: f32,
        eps: f32,
    ) -> Self {
        Optimizer::Adagrad(Adagrad {
            lr,
            wd,
            g_square_momentum,
            initialization,
            eps,
        })
    }

    pub fn new_sgd(lr: f32, wd: f32) -> Self {
        Optimizer::SGD(NaiveSGD { lr, wd })
    }

    pub fn to_optimizable(&self) -> Box<dyn Optimizable + Send + Sync> {
        match &self {
            Optimizer::Adagrad(val) => Box::new(val.clone()) as Box<dyn Optimizable + Send + Sync>,
            Optimizer::SGD(val) => Box::new(val.clone()) as Box<dyn Optimizable + Send + Sync>,
            _ => panic!("not support optimizer: {:?}", &self),
        }
    }
}

pub trait Optimizable {
    fn update(&self, emb_entry: &mut [f32], grad: &[f32], dim: usize);

    #[inline]
    fn require_space(&self, _dim: usize) -> usize {
        0
    }

    // fn update_step(&mut self, feature_space: &str) {}
    fn update_lr(&mut self, lr: f32);

    fn state_initialization(&self, _state: &mut [f32], _dim: usize) {}
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct Adam {
    wd: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    // steps: HashMap<String, f32>,
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct NaiveSGD {
    lr: f32,
    wd: f32,
}

impl Optimizable for NaiveSGD {
    #[inline]
    fn update(&self, emb_entry: &mut [f32], grad: &[f32], _dim: usize) {
        unsafe {
            decayed_sgd_avx2(emb_entry, grad, self.wd, self.lr);
        }
    }

    fn update_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct Adagrad {
    lr: f32,
    wd: f32,
    g_square_momentum: f32,
    initialization: f32,
    eps: f32,
}

impl Optimizable for Adagrad {
    #[inline]
    fn require_space(&self, dim: usize) -> usize {
        dim
    }

    #[inline]
    fn update(&self, emb_entry: &mut [f32], grad: &[f32], dim: usize) {
        let (emb, adagrad_state) = emb_entry.split_at_mut(dim);
        unsafe {
            decayed_adagrad_avx2(
                adagrad_state,
                emb,
                grad,
                self.g_square_momentum,
                self.lr,
                self.eps,
            )
        }
    }

    #[inline]
    fn state_initialization(&self, state: &mut [f32], dim: usize) {
        state[dim..].copy_from_slice(vec![self.initialization; self.require_space(dim)].as_slice());
    }

    fn update_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
