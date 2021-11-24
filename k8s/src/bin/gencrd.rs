use kube::CustomResourceExt;
use persia_operator::crd::PersiaJob;
use std::fs::File;
use std::io::Write;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    output: String,
}

fn main() {
    let args: Cli = Cli::from_args();

    let crd = serde_yaml::to_string(&PersiaJob::crd()).unwrap();

    match File::create(args.output) {
        Ok(mut output) => {
            if let Err(e) = write!(output, "{}", crd) {
                panic!("failed to write file due to {:?}", e);
            }
        }
        Err(e) => {
            panic!("failed to create file due to {:?}", e);
        }
    }
}
