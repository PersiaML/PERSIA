use serde::{Deserialize, Serialize};
use snafu::{  ResultExt, };


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

#[derive(Serialize, Deserialize, Debug)]
pub struct Input {
    pub msg: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Output {}

