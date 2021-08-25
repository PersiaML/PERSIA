import os
import click
import subprocess

from typing import List, Dict

REPLICA_INDEX = int(os.environ.get("JOB_ID", -1))
DEBUG = int(os.environ.get("DEBUG", False))


def generate_dev_env() -> Dict[str, str]:
    env = {}

    # add thirdparty_path into PATH
    thirdparty_path = os.environ.get("THIRDPARTY_PATH", "")
    origin_path = os.environ.get("PATH")
    env["PATH"] = f"{thirdparty_path}:{origin_path}"
    # add thirdparty_path into PYTHONPATH
    python_path = os.environ.get("PYTHONPATH", "")
    origin_pythonpath = os.environ.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{python_path}:{origin_pythonpath}"
    return env


def run_command(cmd: List[str], verb: bool = True):
    if verb:
        cmd_str = " ".join(cmd)
        print(f"execute command: {cmd_str}")

    if DEBUG:
        env = generate_dev_env()
    else:
        env = {}
    subprocess.check_call(cmd, env=env)


@click.command()
@click.argument("filepath", type=str)
@click.option(
    "--gpu-num", type=int, default=1, help="Number of gpu at current replica node"
)
@click.option("--node-rank", type=int, default=0, help="Replica index of trainer")
@click.option("--nnodes", type=int, default=1, help="Replica num of trainer")
def launch_trainer(filepath, gpu_num: int, node_rank: int, nnodes: int):
    cmd = [
        "python3",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        gpu_num,
        "--nnodes",
        nnodes,
        "--node_rank",
        node_rank,
        filepath,
    ]

    run_command(cmd)


@click.command()
@click.argument("filepath", type=str)
@click.option("--replica-index", type=str, help="Replica index of data compose")
@click.option("--replica-size", type=str, help="Replica num of data compose")
def launch_compose(filepath: str, replica_index: int, replica_size: int):
    cmd = [
        "python3",
        "--replica-index",
        replica_index,
        "--replica-size",
        replica_size,
        filepath,
    ]
    run_command(cmd)


@click.command()
@click.option("--port", type=int, help="Middleware server listen port")
@click.option(
    "--global-config", type=str, help="Config of embedding server and middleware"
)
@click.option("--embedding-config", type=str, help="Config of embedding definition")
@click.option("--replica-index", type=str, help="Replica index of middleware")
@click.option("--replica-size", type=str, help="Replica num of middleware")
def launch_middleware(
    port: int,
    global_config: str,
    embedding_config: str,
    replica_index: int,
    replica_size: int,
):
    cmd = [
        "persia-embedding-sharded-middleware",
        "--port",
        port,
        "--global-config",
        global_config,
        "--embeding-config",
        embedding_config,
        "--replica-index",
        replica_index,
        "--replica-size",
        replica_size,
    ]
    run_command(cmd)


@click.command()
@click.option("--port", type=int, help="Embedding server listen port")
@click.option("--replica-index", type=str, help="Replica index of embedding server")
@click.option("--replica-size", type=str, help="Replica num of embedding server")
@click.option(
    "--global-config", type=str, help="Config of embedding server and middleware"
)
def launch_server(port: int, replica_index: int, replica_size: int, global_config: str):
    cmd = [
        "persia-embedding-sharded-server",
        "--port",
        port,
        "--global-config",
        global_config,
        "--shard-idx",
        replica_index,
        "--num-shards",
        replica_size,
    ]
    subprocess.check_call(cmd, shell=True)


@click.command()
def launch_local():
    # TODO: launch the trainer, middleware server, embedding server
    cmd = "launch the trainer locally with middleware and embedding server"
    print(cmd)
