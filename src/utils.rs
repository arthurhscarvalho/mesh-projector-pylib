use numpy::PyReadonlyArray2;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

pub fn build_thread_pool(max_workers: usize) -> ThreadPool {
    let pool = ThreadPoolBuilder::new()
        .num_threads(max_workers)
        .build()
        .expect("Failed to build thread pool");
    pool
}
