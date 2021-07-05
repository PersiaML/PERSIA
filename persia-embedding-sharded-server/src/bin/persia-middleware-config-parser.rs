#[macro_use]
extern crate shadow_rs;
use persia_embedding_sharded_server::middleware_config_parser::{MiddlewareConfig, convert_middleware_config};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    config: PathBuf,
    #[structopt(long)]
    new_config: PathBuf,
}

fn main() -> anyhow::Result<()> {
    color_eyre::install().unwrap();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();
    shadow!(build);
    eprintln!("project_name: {}", build::PROJECT_NAME);
    eprintln!("is_debug: {}", shadow_rs::is_debug());
    eprintln!("version: {}", build::version());
    eprintln!("tag: {}", build::TAG);
    eprintln!("commit_hash: {}", build::COMMIT_HASH);
    eprintln!("commit_date: {}", build::COMMIT_DATE);
    eprintln!("build_os: {}", build::BUILD_OS);
    eprintln!("rust_version: {}", build::RUST_VERSION);
    eprintln!("build_time: {}", build::BUILD_TIME);
    let args: Cli = Cli::from_args();

    let mut config: MiddlewareConfig =
        serde_yaml::from_reader(std::fs::File::open(args.config).expect("cannot read config file"))
            .expect("cannot parse config file");
    
    let _ = convert_middleware_config(&mut config);

    serde_yaml::to_writer(
        std::fs::File::create(args.new_config)
            .expect("cannot create new config file"),
        &config)
        .expect("failed to write new config file");

    Ok(())
}