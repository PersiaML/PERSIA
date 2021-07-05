#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use criterion::*;
use criterion_macro::criterion;

extern crate persia_embedding_sharded_server;
use persia_embedding_sharded_server::dev_utils::{
    init_sharded_server, init_middleware, gen_sharded_server_request,
    gen_middleware_request,
};
use std::time::Duration;

const THREADS: usize = 40;

#[criterion]
fn bench_sharded_server(c: &mut Criterion) {
    let (sharded_server_client, runtime, mut sharded_server) = init_sharded_server();
    let (middleware_client, mut middleware) = init_middleware();

    let mut group = c.benchmark_group("end2end");
    group.throughput(Throughput::Elements(1));

    group.bench_function("sharded_server_b2560", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let sharded_server_client = sharded_server_client.clone();                    
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let x = gen_sharded_server_request(2560);
                            let lookup_req = x.0;
                            let update_grad_req = (x.1, x.2);
                            let start = std::time::Instant::now();
                            let result =
                                runtime.block_on(sharded_server_client.lookup_mixed(black_box(&lookup_req)));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(sharded_server_client.update_gradient_mixed(black_box(&update_grad_req)));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("sharded_server_b2560_compressed", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let sharded_server_client = sharded_server_client.clone();                    
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let x = gen_sharded_server_request(2560);
                            let lookup_req = x.0;
                            let update_grad_req = (x.1, x.2);
                            let start = std::time::Instant::now();
                            let result =
                                runtime.block_on(sharded_server_client.lookup_mixed_compressed(black_box(&lookup_req)));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(sharded_server_client.update_gradient_mixed_compressed(black_box(&update_grad_req)));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("sharded_server_b2560_compressed_lookup", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let sharded_server_client = sharded_server_client.clone();                    
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let x = gen_sharded_server_request(2560);
                            let lookup_req = x.0;
                            let update_grad_req = (x.1, x.2);
                            let start = std::time::Instant::now();
                            let result =
                                runtime.block_on(sharded_server_client.lookup_mixed_compressed(black_box(&lookup_req)));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(sharded_server_client.update_gradient_mixed(black_box(&update_grad_req)));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("middleware_b2560", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let middleware_client = middleware_client.clone();
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let (sparse_batch, grad_batch) = gen_middleware_request(2560);
                            let start = std::time::Instant::now();
                            let forward_id =
                                runtime.block_on(middleware_client.forward_batched(black_box(&sparse_batch)))
                                .unwrap()
                                .unwrap();
                            let result =
                                runtime.block_on(middleware_client.forward_batch_id(black_box(&(forward_id, false))));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(middleware_client.update_gradient_batched(black_box(&(forward_id, grad_batch))));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("sharded_server_b256", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let sharded_server_client = sharded_server_client.clone();                    
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let x = gen_sharded_server_request(256);
                            let lookup_req = x.0;
                            let update_grad_req = (x.1, x.2);
                            let start = std::time::Instant::now();
                            let result =
                                runtime.block_on(sharded_server_client.lookup_mixed(black_box(&lookup_req)));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(sharded_server_client.update_gradient_mixed(black_box(&update_grad_req)));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("sharded_server_b256_compressed", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let sharded_server_client = sharded_server_client.clone();                    
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let x = gen_sharded_server_request(256);
                            let lookup_req = x.0;
                            let update_grad_req = (x.1, x.2);
                            let start = std::time::Instant::now();
                            let result =
                                runtime.block_on(sharded_server_client.lookup_mixed_compressed(black_box(&lookup_req)));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(sharded_server_client.update_gradient_mixed_compressed(black_box(&update_grad_req)));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("sharded_server_b256_compressed_lookup", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let sharded_server_client = sharded_server_client.clone();                    
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let x = gen_sharded_server_request(256);
                            let lookup_req = x.0;
                            let update_grad_req = (x.1, x.2);
                            let start = std::time::Instant::now();
                            let result =
                                runtime.block_on(sharded_server_client.lookup_mixed_compressed(black_box(&lookup_req)));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(sharded_server_client.update_gradient_mixed(black_box(&update_grad_req)));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    group.bench_function("middleware_b256", |b| {
        b.iter_custom(|iters| {
            let mut join = Vec::with_capacity(THREADS);
            join.extend((0..THREADS).map(|_| {
                std::thread::spawn({
                    let runtime = runtime.clone();
                    let middleware_client = middleware_client.clone();
                    move || {
                        let mut dur = Duration::new(0, 0);
                        let _guard = runtime.enter();
                        for _ in 0..iters {
                            let (sparse_batch, grad_batch) = gen_middleware_request(256);
                            let start = std::time::Instant::now();
                            let forward_id =
                                runtime.block_on(middleware_client.forward_batched(black_box(&sparse_batch)))
                                .unwrap()
                                .unwrap();
                            let result =
                                runtime.block_on(middleware_client.forward_batch_id(black_box(&(forward_id, false))));
                            assert_eq!(result.is_ok(), true);
                            let result =
                                runtime.block_on(middleware_client.update_gradient_batched(black_box(&(forward_id, grad_batch))));
                            assert_eq!(result.is_ok(), true);
                            dur = dur.checked_add(start.elapsed()).unwrap();
                        }
                        dur
                    }
                })
            }));

            let durations: Vec<Duration> = join
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();

            let sum: Duration = durations.iter().sum();
            sum / (durations.len() as u32)
            // durations.sort_unstable();
            // durations[durations.len() / 2]
        });
    });

    while sharded_server.kill().is_err() {
        tracing::info!("retrying to kill sharded server...")
    }
    while middleware.kill().is_err() {
        tracing::info!("retrying to kill middleware...")
    }
}
