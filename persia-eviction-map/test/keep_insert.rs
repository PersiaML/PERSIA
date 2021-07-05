use persia_eviction_map::PersiaEvictionMap;

use persia_embedding_config::InitializationMethod;
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use std::{sync::Arc, u64};
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    capacity: usize,
}

// type ArcEntry = Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>;

fn main() -> () {

    let args: Cli = Cli::from_args();
    let cap = args.capacity;

    let map = Arc::new(PersiaEvictionMap::new(cap, 512));
    for _ in 0..10 {
        let map = map.clone();
        let _handle = std::thread::spawn(move || {
            let initialization = InitializationMethod::default();
            loop {
                for i in 0..cap {
                    let entry = HashMapEmbeddingEntry::new(&initialization, 0.1, 64, i as u64);
                    map.insert(i as u64, Arc::new(parking_lot::RwLock::new(entry)));
                }
                println!("finish one loop");
            }
        });
    }

    let initialization = InitializationMethod::default();
    loop {
        for i in 0..1000 {
            let entry = HashMapEmbeddingEntry::new(&initialization, 0.1, 64, i as u64);
            map.insert(i, Arc::new(parking_lot::RwLock::new(entry)));
        }
    }
}

// fn main() -> () {
//     let map = Arc::new(PersiaEvictionMap::new(50000000, 512));
//     for _ in 0..10 {
//         let map = map.clone();
//         let handle = std::thread::spawn(move || {
//             loop {
//                 for i in 0..50000000 {
//                     map.insert(i as u64, i as u64);
//                 }
//                 println!("finish one loop");
//             }
//         });
//     }

//     let initialization = InitializationMethod::default();
//     loop {
//         for i in 0..1000 {
//             map.insert(i as u64, i as u64);
//         }
//     }
// }


// fn main() -> () {

//     let args: Cli = Cli::from_args();
//     let cap = args.capacity;

//     let map = Arc::new(PersiaEvictionMap::new(cap, 512));
//     for _ in 0..10 {
//         let map = map.clone();
//         let handle = std::thread::spawn(move || {
//             let initialization = InitializationMethod::default();
//             loop {
//                 for i in 0..cap {
//                     let entry = HashMapEmbeddingEntry::new(&initialization, 0.1, 64, i as u64);
//                     map.insert(i, entry);
//                 }
//                 println!("finish one loop");
//             }
//         });
//     }

//     let initialization = InitializationMethod::default();
//     loop {
//         for i in 0..1000 {
//             let entry = HashMapEmbeddingEntry::new(&initialization, 0.1, 64, i as u64);
//             map.insert(i, entry);
//         }
//     }
// }