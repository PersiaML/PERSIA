use snafu::{ResultExt};
use persia_speedy::{Readable, Writable};


#[derive(Clone)]
pub struct Service {}

#[persia_rpc::service]
impl Service {
    pub async fn rpc_test(&self, input: Input) -> Output {
        dbg!(input);
        return Output {};
    }

    pub async fn rpc_test_2(&self, input: ()) -> Output {
        return Output {};
    }

    pub async fn large_body_rpc_test(&self, input: Vec<f32>) -> Vec<f32> {
        vec![0.; 20971520]
    }
}

#[derive(Readable, Writable, Debug)]
pub struct Input {
    pub msg: String,
}

#[derive(Readable, Writable, Debug)]
pub struct Output {}

#[derive(Readable, Writable, Debug)]
pub struct RecyclableVec(Vec<f32>);

impl RecyclableVec {

}