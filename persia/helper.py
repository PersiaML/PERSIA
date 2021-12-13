import os
import yaml
import subprocess
import tempfile
import cloudpickle

from typing import List, Callable, Optional
from contextlib import contextmanager

from persia.utils import resolve_binary_execute_path


_CLOUD_PICKLE_TEMPLATE = """
import cloudpickle
def main():
    remote_run_function = cloudpickle.load(open("{}", "rb"))
    remote_run_function()
if __name__ == "__main__":
    main()
"""


def _launch_serve(
    server_type: str, replica_num: int, env: os._Environ, port: int = 8888
) -> List[subprocess.Popen]:

    processes = []
    executable_path = resolve_binary_execute_path(server_type)
    for replica_idx in range(replica_num):
        cmd = [
            executable_path,
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
        current_env = env.copy()
        current_env["REPLICA_SIZE"] = str(replica_num)
        current_env["REPLICA_INDEX"] = str(replica_index)

        cmd = ["python3", env["PERSIA_DATALOADER_ENTRY"]]

        processes.append(
            subprocess.Popen(
                cmd,
                env=current_env,
            )
        )
    return processes


def _launch_nn_worker(nproc_per_node: int, env: os._Environ) -> subprocess.Popen:
    return subprocess.Popen(
        ["persia-launcher", "nn-worker", "--nproc-per-node", str(nproc_per_node)],
        env=env,
    )


def _launch_nats_server() -> List[subprocess.Popen]:
    return [subprocess.Popen(["nats-server"])]


@contextmanager
def ensure_persia_service(
    data_loader_func: Optional[Callable] = None,
    nn_worker_func: Optional[Callable] = None,
    embedding_config: Optional[dict] = None,
    embedding_config_path: str = None,
    global_config: Optional[dict] = None,
    global_config_path: Optional[str] = None,
    data_laoder_replica_num: Optional[int] = 1,
    nproc_per_node: Optional[int] = 1,
    embedding_worker_replica_num: Optional[int] = 1,
    embedding_parameter_replica_num: Optional[int] = 1,
    embedding_worker_port: Optional[int] = 7777,
    embedding_parameter_server_port: Optional[int] = 8888,
    persia_nats_url: Optional[str] = None,
):
    process_group: List[subprocess.Popen] = []

    with tempfile.NamedTemporaryFile(
        "w"
    ) as embedding_config_file, tempfile.NamedTemporaryFile(
        "w"
    ) as global_config_file, tempfile.NamedTemporaryFile(
        "wb"
    ) as cloudpickle_file, tempfile.NamedTemporaryFile(
        "w"
    ) as python_file:
        if not embedding_config_path:
            assert embedding_config is not None
            embedding_config_file.write(yaml.dump(embedding_config))
            embedding_config_file.flush()
            embedding_config_path = embedding_config_file.name

        if not global_config_path:
            assert global_config is not None
            global_config_file.write(yaml.dump(global_config))
            global_config_file.flush()
            global_config_path = global_config_file.name

        _ENV = os.environ
        _ENV["PERSIA_EMBEDDING_CONFIG"] = embedding_config_path
        _ENV["PERSIA_GLOBAL_CONFIG"] = global_config_path
        _ENV["PERSIA_NATS_URL"] = persia_nats_url or "localhost:4222"
        _ENV["LOG_LEVEL"] = "info"

        if data_loader_func is not None:
            pickle_data_loader = cloudpickle.dumps(data_loader_func)
            cloudpickle_file.write(pickle_data_loader)
            python_file.write(_CLOUD_PICKLE_TEMPLATE.format(cloudpickle_file.name))

            _ENV["PERSIA_DATALOADER_ENTRY"] = python_file.name
            cloudpickle_file.flush()
            python_file.flush()
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
                "persia-embedding-worker",
                replica_num=embedding_worker_replica_num,
                env=_ENV,
                port=embedding_worker_port,
            )
        )
        # add embedding-parameter-server
        process_group.extend(
            _launch_serve(
                "persia-embedding-parameter-server",
                replica_num=embedding_parameter_replica_num,
                env=_ENV,
                port=embedding_parameter_server_port,
            )
        )

        """Launch three service, embedding_server, """
        yield

        for process in process_group:
            process.kill()

        for process in process_group:
            process.wait()


if __name__ == "__main__":

    def data_loader():
        import numpy as np

        from persia.ctx import DataCtx
        from persia.embedding.data import PersiaBatch, IDTypeFeature
        from persia.logger import get_default_logger

        _logger = get_default_logger()

        persia_batch = PersiaBatch(
            [
                IDTypeFeature(
                    "age",
                    [
                        np.array(
                            [
                                1,
                                2,
                                3,
                            ],
                            dtype=np.uint64,
                        )
                    ],
                )
            ],
            requires_grad=False,
        )

        with DataCtx() as data_ctx:
            data_ctx.send_data(persia_batch)
            _logger.info("send msg done.")

    embedding_config = {"slots_config": {"age": {"dim": 8}}}
    global_config = {
        "embedding_worker_config": {"forward_buffer_size": 1000},
        "common_config": {"metrics_config": {"enable_metrics": False}},
    }

    os.environ["WORLD_SIZE"] = str(1)
    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)

    from persia.ctx import BaseCtx
    from persia.data import Dataloder, StreamingDataset
    from persia.embedding import get_default_embedding_config
    from persia.env import get_world_size
    from persia.logger import get_default_logger

    _logger = get_default_logger()

    with ensure_persia_service(
        data_loader_func=data_loader,
        embedding_config=embedding_config,
        global_config=global_config,
    ):
        embedding_config = get_default_embedding_config()

        with BaseCtx() as ctx:
            ctx.common_context.init_nats_publisher(get_world_size())
            ctx.common_context.configure_embedding_parameter_servers(
                embedding_config.emb_initialization[0],
                embedding_config.emb_initialization[1],
                embedding_config.admit_probability,
                embedding_config.weight_bound > 0,
                embedding_config.weight_bound,
            )
            ctx.common_context.wait_servers_ready()

            data_loader = Dataloder(StreamingDataset(buffer_size=15))
            data_generator = iter(data_loader)
            data = next(data_generator)

            _logger.info("get data done...")
