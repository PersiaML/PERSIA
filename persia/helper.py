import os
import yaml
import subprocess
import tempfile
import cloudpickle

from typing import List, Callable, Optional
from contextlib import contextmanager


_DEFAULT_EMBEDDING_CONFIG = {"prefix_index": 10, "slot_configs": []}

_DEFAULT_GLOBLA_CONFIG = {}

_CLOUD_PICKLE_TEMPLATE = """
import cloudpickle
def main():
    remote_run_function = cloudpickle.load({})
    remote_run_function()
if __name__ == "__main__":
    main()
"""


def _launch_serve(
    server_type: str, replica_num: int, env: os._Environ, port: int = 8888
) -> List[subprocess.Popen]:

    processes = []
    for replica_idx in range(replica_num):
        cmd = [
            "persia-launcher",
            server_type,
            "--replica-size",
            str(replica_num),
            "--replica-index",
            str(replica_idx),
            "--port",
            str(port + replica_idx),
        ]
        processes.append(subprocess.Popen(cmd, env=env))

    return processes


def _launch_data_loader(replica_num: int, env: os._Environ) -> List[subprocess.Popen]:
    processes = []
    for replica_index in range(replica_num):
        cmd = [
            "persia-launcher",
            "data-loader",
            "--replica-size",
            str(replica_num),
            "--replica-index",
            str(replica_index),
        ]

        processes.append(subprocess.Popen(cmd, env=env))
    return processes


def _launch_nn_worker(nproc_per_node: int, env: os._Environ) -> subprocess.Popen:
    return subprocess.Popen(
        ["persia-launcher", "nn-worker", "--nproc-per-node", str(nproc_per_node)],
        env=env,
    )


def _launch_nats_server() -> List[subprocess.Popen]:
    return [subprocess.Popen(["nats-server"])]


@contextmanager
def ensure_persia_env(
    data_loader_func: Optional[Callable] = None,
    nn_worker_func: Optional[Callable] = None,
    slot_configs: Optional[List[dict]] = None,
    data_laoder_replica_num: Optional[int] = 1,
    nproc_per_node: Optional[int] = 1,
    embedding_worker_replica_num: Optional[int] = 1,
    embedding_parameter_replica_num: Optional[int] = 1,
    persia_nats_url: Optional[str] = None,
):
    process_group = []
    named_tempfiles = []

    with tempfile.NamedTemporaryFile(
        "w+"
    ) as embedding_config, tempfile.NamedTemporaryFile(
        "w+"
    ) as global_config, tempfile.NamedTemporaryFile(
        "wb+"
    ) as cloudpickle_file, tempfile.NamedTemporaryFile(
        "w+"
    ) as python_file:
        _DEFAULT_EMBEDDING_CONFIG["slot_configs"].extend(slot_configs)
        embedding_config.write(yaml.dump(_DEFAULT_EMBEDDING_CONFIG))
        global_config.write(yaml.dump(_DEFAULT_GLOBLA_CONFIG))

        _ENV = os.environ
        _ENV["PERSIA_EMBEDDING_CONFIG"] = embedding_config.name
        _ENV["PERSIA_GLOBAL_CONFIG"] = global_config.name
        _ENV["PERSIA_NATS_URL"] = persia_nats_url or "localhost:4222"

        if data_loader_func is not None:

            pickle_data_loader = cloudpickle.dumps(data_loader_func)
            cloudpickle_file.write(pickle_data_loader)
            python_file.write(_CLOUD_PICKLE_TEMPLATE.format(cloudpickle_file.name))

            _ENV["PERSIA_DATALOADER_ENTRY"] = python_file.name
            process_group.extend(_launch_data_loader(data_laoder_replica_num, _ENV))
        else:
            pickle_nn_worker = cloudpickle.dumps(nn_worker_func)
            cloudpickle_file.write(pickle_nn_worker)
            python_file.write(_CLOUD_PICKLE_TEMPLATE.format(cloudpickle_file.name))

            _ENV["PERSIA_NN_WORKER_ENTRY"] = python_file.name
            process_group.append(_launch_nn_worker(nproc_per_node, _ENV))

        process_group.extend(_launch_nats_server())
        # add embedding-worker
        process_group.extend(
            _launch_serve(
                "embedding-worker", replica_num=embedding_worker_replica_num, env=_ENV
            )
        )
        # add embedding-parameter-server
        process_group.extend(
            _launch_serve(
                "embedding-parameter-worker",
                replica_num=embedding_parameter_replica_num,
                env=_ENV,
            )
        )

        """Launch three service, embedding_server, """
        yield

        for process in process_group:
            process.kill()