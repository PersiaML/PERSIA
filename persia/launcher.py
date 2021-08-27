import os
import click
import subprocess

from typing import List

from persia.logger import get_logger

_logger = get_logger("launcher")

REPLICA_INDEX = int(os.environ.get("JOB_ID", -1))
DEBUG = int(os.environ.get("DEBUG", False))

_ENV = os.environ.copy()

if DEBUG:
    # add thirdparty_path into PATH
    thirdparty_path = os.environ.get("THIRDPARTY_PATH", "")
    origin_path = os.environ.get("PATH")
    _ENV["PATH"] = f"{thirdparty_path}:{origin_path}"
    # add thirdparty_path into PYTHONPATH
    python_path = os.environ.get("PYTHONPATH", "")
    origin_pythonpath = os.environ.get("PYTHONPATH")
    _ENV["PYTHONPATH"] = f"{python_path}:{origin_pythonpath}"


def resolve_binary_execute_path(binary_name: str) -> str:
    if DEBUG:
        thirdparty_path = os.environ.get("THIRDPARTY_PATH", None)
        if not thirdparty_path:
            raise KeyError(
                "Launch program with debug mode but without THIRDPARTY_PATH env"
            )

        if os.access(os.path.join(thirdparty_path, binary_name), os.X_OK):
            raise Exception(
                f"Can't not found executable {binary_name} in {thirdparty_path}"
            )

        return binary_name
    else:
        return os.path.realpath(os.path.join(__file__, "../", binary_name))


def run_command(cmd: List[str], verb: bool = True):
    cmd = list(map(str, cmd))
    if verb:
        cmd_str = " ".join(cmd)
        print(f"execute command: {cmd_str}")

    subprocess.check_call(cmd, env=_ENV)


@click.group()
def cli():
    ...


@cli.command()
@click.argument("filepath", type=str)
@click.option(
    "--gpu-num", type=int, default=1, help="Number of gpu at current replica node"
)
@click.option("--node-rank", type=int, default=0, help="Replica index of trainer")
@click.option("--nnodes", type=int, default=1, help="Replica num of trainer")
def trainer(filepath, gpu_num: int, node_rank: int, nnodes: int):
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


@cli.command()
@click.argument("filepath", type=str)
@click.option(
    "--replica-index", type=str, default=0, help="Replica index of data compose"
)
@click.option("--replica-size", type=str, default=1, help="Replica num of data compose")
def compose(filepath: str, replica_index: int, replica_size: int):
    cmd = [
        "python3",
        filepath,
        "--replica-index",
        replica_index,
        "--replica-size",
        replica_size,
    ]
    run_command(cmd)


@cli.command()
@click.option("--port", type=int, default=8887, help="Middleware server listen port")
@click.option("--embedding-config", type=str, help="Config of embedding definition")
@click.option(
    "--global-config", type=str, help="Config of embedding server and middleware"
)
@click.option(
    "--replica-index", type=str, default=0, help="Replica index of middleware"
)
@click.option("--replica-size", type=str, default=1, help="Replica num of middleware")
def middleware(
    port: int,
    embedding_config: str,
    global_config: str,
    replica_index: int,
    replica_size: int,
):
    executable_path = resolve_binary_execute_path("persia-embedding-sharded-middleware")
    cmd = [
        executable_path,
        "--port",
        port,
        "--global-config",
        global_config,
        "--embedding-config",
        embedding_config,
        "--replica-index",
        replica_index,
        "--replica-size",
        replica_size,
    ]
    run_command(cmd)


@cli.command()
@click.option("--port", type=int, default=8888, help="Embedding server listen port")
@click.option("--embedding-config", type=str, help="Config of embedding definition")
@click.option(
    "--global-config", type=str, help="Config of embedding server and middleware"
)
@click.option(
    "--replica-index", type=str, default=0, help="Replica index of embedding server"
)
@click.option(
    "--replica-size", type=str, default=1, help="Replica num of embedding server"
)
def server(
    port: int,
    embedding_config: str,
    global_config: str,
    replica_index: int,
    replica_size: int,
):
    executable_path = resolve_binary_execute_path("persia-embedding-sharded-server")
    cmd = [
        executable_path,
        "--port",
        port,
        "--global-config",
        global_config,
        "--embedding-config",
        embedding_config,
        "--shard-idx",
        replica_index,
        "--num-shards",
        replica_size,
    ]
    run_command(cmd)


@cli.command()
def trainer_local():
    # TODO: launch the trainer, middleware server, embedding server
    cmd = "launch the trainer locally with middleware and embedding server"
    print(cmd)


if __name__ == "__main__":
    cli()
