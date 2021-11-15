use kube::CustomResourceExt;
use persia_operator::crd::PersiaJob;
fn main() {
    print!("{}", serde_yaml::to_string(&PersiaJob::crd()).unwrap());
}
