import os
import click
import subprocess

from typing import List

from persia.logger import get_logger

_logger = get_logger(__file__)

_ENV = os.environ.copy()


def resolve_binary_execute_path(binary_name: str) -> str:
    return os.path.realpath(os.path.join(__file__, "../", binary_name))


def run_command(cmd: List[str], verb: bool = True):
    cmd = list(map(str, cmd))
    if verb:
        cmd_str = " ".join(cmd)
        _logger.info(f"execute command: {cmd_str}")

    subprocess.check_call(cmd, env=_ENV)


@click.group()
def cli():
    ...


@cli.command()
@click.argument("filepath", type=str)
@click.option("--nproc-per-node", type=int, default=1, help="Process num of per node")
@click.option("--node-rank", type=int, default=0, help="Replica index of nn worker")
@click.option("--nnodes", type=int, default=1, help="Replica num of nn owrker")
def nn_worker(filepath, nproc_per_node: int, node_rank: int, nnodes: int):

    cmd = [
        "python3",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        nproc_per_node,
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
    "--replica-index", type=str, default=0, help="Replica index of data loader"
)
@click.option("--replica-size", type=str, default=1, help="Replica num of data loader")
def data_loader(filepath: str, replica_index: int, replica_size: int):
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
@click.option("--port", type=int, default=8887, help="Embedding worker listen port")
@click.option("--embedding-config", type=str, help="Config of embedding definition")
@click.option(
    "--global-config", type=str, help="Config of embedding server and embedding worker"
)
@click.option(
    "--replica-index", type=str, default=0, help="Replica index of embedding worker"
)
@click.option(
    "--replica-size", type=str, default=1, help="Replica num of embedding worker"
)
def embedding_worker(
    port: int,
    embedding_config: str,
    global_config: str,
    replica_index: int,
    replica_size: int,
):
    executable_path = resolve_binary_execute_path("persia-embedding-worker")
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
    "--global-config", type=str, help="Config of embedding server and embedding worker"
)
@click.option(
    "--replica-index", type=str, default=0, help="Replica index of embedding server"
)
@click.option(
    "--replica-size", type=str, default=1, help="Replica num of embedding server"
)
def embedding_parameter_server(
    port: int,
    embedding_config: str,
    global_config: str,
    replica_index: int,
    replica_size: int,
):
    executable_path = resolve_binary_execute_path("persia-embedding-parameter-server")
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


if __name__ == "__main__":
    cli()
