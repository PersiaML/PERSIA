use snafu::{  ResultExt, };
use persia_speedy::{Readable, Writable};


#[derive(Clone)]
pub struct Service {

}

#[persia_rpc::service]
impl Service {
    pub async fn rpc_test(&self, input: Input) -> Output {
        dbg!(input);
        return Output {};
    }

    pub async fn rpc_test_2(&self) -> Output {
        return Output {};
    }
}

#[derive(Readable, Writable, Debug)]
pub struct Input {
    pub msg: String,
}

#[derive(Readable, Writable, Debug)]
pub struct Output {}

