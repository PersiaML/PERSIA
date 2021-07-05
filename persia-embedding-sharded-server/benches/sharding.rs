#![feature(test)]
extern crate test;

use random_fast_rng::Random;
use seahash::SeaHasher;
use std::hash::Hasher;
use wyhash::WyHash;

#[bench]
fn to_bytes(b: &mut test::Bencher) {
    let sign = random_fast_rng::local_rng().get_u64();
    b.iter(|| {
        std::hint::black_box(&sign.to_le_bytes());
    })
}

#[bench]
fn clone_u64(b: &mut test::Bencher) {
    let sign = random_fast_rng::local_rng().get_u64();
    b.iter(|| {
        std::hint::black_box(sign.clone());
    })
}

#[bench]
fn multiply(b: &mut test::Bencher) {
    let mut sign = random_fast_rng::local_rng().get_u64();
    b.iter(|| {
        sign = sign.wrapping_mul(11400714819323198549u64);
    })
}

#[bench]
fn farmhash(b: &mut test::Bencher) {
    let mut sign = random_fast_rng::local_rng().get_u64();
    b.iter(|| {
        sign = farmhash::hash64(&sign.to_le_bytes());
    })
}

#[bench]
fn wyhash(b: &mut test::Bencher) {
    let mut sign = random_fast_rng::local_rng().get_u64();
    let mut hasher = WyHash::with_seed(5);
    b.iter(|| {
        hasher.write_u64(sign.clone());
        sign = hasher.finish();
    })
}

#[bench]
fn seahash(b: &mut test::Bencher) {
    let mut sign = random_fast_rng::local_rng().get_u64();
    let mut hasher = SeaHasher::new();
    b.iter(|| {
        hasher.write_u64(sign.clone());
        sign = hasher.finish();
    })
}
