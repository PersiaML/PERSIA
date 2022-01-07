r"""
``persia.launcher`` is a module that provides the functionality to help user launch the PERSIA
service. A command line interface ``persia-launcher`` is installed automatically
when you install PERSIA. You can launch the PERSIA modules ``data-loader``, ``nn-worker``,
``embedding-worker``, and ``embedding-parameter-server`` by invoking the subcommand from
``persia-launcher``.

.. code-block:: bash

    persia-launcher --help

Or

.. code-block:: bash

    python3 -m persia.launcher --help

=================
NN Worker And Data Loader
=================

If you want to launch the ``nn-worker`` and ``data-loader``, you can use:

.. code-block:: bash

    # Launch the nn-worker
    persia-launcher nn-worker train.py --nproc-per-node 1 --node-rank 0 --nnodes 1

    # Launch the data-loader
    persia-launcher data-loader data_loader.py --replica-size 1 --replica-index 0


=================
Embedding Worker And Embedding Parameter Server
=================

To launch the ``embedding-worker`` and ``embedding-parameter-server``, you can use:

.. code-block:: bash

    # Launch the embedding-worker
    persia-launcher embedding-worker \
        --global-config global_config.yml \
        --embedding-config embedding_config.yml \
        --replica-index 0 --replica-size 1

    # Launch the embedding-parameter-server
    persia-launcher embedding-parameter-server\
        --global-config global_config.yml \
        --embedding-config embedding_config.yml \
        --replica-index 0 --replica-size 1

=================
Arguments from Environment Variables
=================

Some arguments will fallback to the environment variable if the
arguments are not provided when running the ``persia-launcher``.
It is useful to control the environment when you deploy the PERSIA
job in container management tool.

For ``nn-worker``, you can export the environment variable
``PERSIA_NN_WORKER_ENTRY`` to avoid passing the filepath argument when launching the
`nn-worker`. And the environment variable ``PERSIA_DATALOADER_ENTRY`` is for
``data-loader``.

.. code-block::

    export PERSIA_DATALOADER_ENTRY=data_loader.py

    # Launch the data_load
    persia-launcher data-loader --replica-index 0 --replica-size 1

    export PERSIA_NN_WORKER_ENTRY=train.py

    # Launch the nn-worker
    persia-launcher nn-worker --nproc-per-node 1 --node-rank 0 --nnodes 1

For ``embedding-worker`` and ``embedding-parameter-server``, you can export the environment
variable ``PERSIA_EMBEDDING_CONFIG`` and ``PERSIA_GLOBAL_CONFIG`` to avoid passing the
config filepath to cli command.

.. code-block::

    export PERSIA_EMBEDDING_CONFIG=embedding_config.yml
    export PERSIA_GLOBAL_CONFIG=global_config.yml

    # Launch the embedding-worker
    persia-launcher embedding-worker \
        --replica-index 0 --replica-size 1

    # Launch the embedding-parameter-server
    persia-launcher embedding-parameter-server \
        --replica-index 0 --replica-size 1

"""

import os
import click

from persia.utils import run_command, resolve_binary_execute_path

_ENV = os.environ.copy()


@click.group()
def cli():
    ...


@cli.command()
@click.argument(
    "filepath",
    envvar="PERSIA_NN_WORKER_ENTRY",
    type=str,
)
@click.option("--nproc-per-node", type=int, default=1, help="Process num of per node")
@click.option("--node-rank", type=int, default=0, help="Replica index of nn worker")
@click.option("--nnodes", type=int, default=1, help="Replica num of nn owrker")
def nn_worker(filepath: str, nproc_per_node: int, node_rank: int, nnodes: int):
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
    run_command(cmd, _ENV)


@cli.command()
@click.argument(
    "filepath",
    envvar="PERSIA_DATALOADER_ENTRY",
    type=str,
)
@click.option(
    "--replica-index", type=str, default="0", help="Replica index of data loader"
)
@click.option(
    "--replica-size", type=str, default="1", help="Replica num of data loader"
)
def data_loader(filepath: str, replica_index: int, replica_size: int):
    _ENV["REPLICA_INDEX"] = replica_index
    _ENV["REPLICA_SIZE"] = replica_size

    cmd = [
        "python3",
        filepath,
    ]
    run_command(cmd, _ENV)


@cli.command()
@click.option("--port", type=int, default=8887, help="Embedding worker listen port")
@click.option(
    "--embedding-config",
    envvar="PERSIA_EMBEDDING_CONFIG",
    help="Config path of embedding definition. Use PERSIA_EMBEDDING_CONFIG environment \
        as the default value if not passed the value to CLI command",
)
@click.option(
    "--global-config",
    type=str,
    envvar="PERSIA_GLOBAL_CONFIG",
    help="Config of embedding server and embedding worker. Use PERSIA_GLOBAL_CONFIG environment \
        as the default value if not passed the value to CLI command",
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
    run_command(cmd, _ENV)


@cli.command()
@click.option("--port", type=int, default=8888, help="Embedding server listen port")
@click.option(
    "--embedding-config",
    type=str,
    envvar="PERSIA_EMBEDDING_CONFIG",
    help="Config of embedding definition. Use PERSIA_EMBEDDING_CONFIG environment \
        as the default value if not passed the value to CLI command",
)
@click.option(
    "--global-config",
    envvar="PERSIA_GLOBAL_CONFIG",
    type=str,
    help="Config of embedding server and embedding worker. Use PERSIA_EMBEDDING_CONFIG environment \
        as the default value if not passed the value to CLI command",
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
    run_command(cmd, _ENV)


if __name__ == "__main__":
    cli()
