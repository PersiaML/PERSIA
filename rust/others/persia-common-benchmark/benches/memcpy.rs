#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use criterion::*;
use criterion_macro::criterion;

fn memcpy(bytes_src: &[u8], bytes_dst: &mut [u8]) {
    unsafe {
        std::ptr::copy(bytes_src.as_ptr(), bytes_dst.as_mut_ptr(), bytes_src.len());
    }
}

#[criterion]
fn bench_memcpy(c: &mut Criterion) {
    let bytes_src = vec![0; 1024 * 1024 * 5]; // 128 MB
    let mut bytes_dst = vec![0; 1024 * 1024 * 5]; // 128 MB
    let mut group = c.benchmark_group("memcpy");
    group.throughput(Throughput::Bytes(bytes_src.len() as u64));
    group.bench_function("memcpy", |b| {
        b.iter(|| {
            memcpy(
                black_box(bytes_src.as_slice()),
                black_box(bytes_dst.as_mut_slice()),
            )
        })
    });
    group.finish();
}
