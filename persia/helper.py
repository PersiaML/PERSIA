import os
import yaml
import subprocess
import time
import tempfile
import cloudpickle

from typing import List, Callable, Optional, Tuple, IO, Union
from threading import Thread, Lock

from persia.utils import resolve_binary_execute_path, find_free_port


_CLOUD_PICKLE_TEMPLATE = """
import cloudpickle
def main():
    remote_run_function = cloudpickle.load(open("{}", "rb"))
    remote_run_function()
if __name__ == "__main__":
    main()
"""

_DEFAULT_GLOBAL_CONFIG = {
    "embedding_worker_config": {"forward_buffer_size": 1000},
    "common_config": {"metrics_config": {"enable_metrics": False}},
}


def _dump_function_into_tempfile(
    func: Callable,
) -> Tuple[IO[bytes], IO[str]]:
    cloudpickle_file, python_file = (
        tempfile.NamedTemporaryFile("wb"),
        tempfile.NamedTemporaryFile("w"),
    )
    pickle_data_loader = cloudpickle.dumps(func)
    cloudpickle_file.write(pickle_data_loader)
    python_file.write(_CLOUD_PICKLE_TEMPLATE.format(cloudpickle_file.name))
    cloudpickle_file.flush()
    python_file.flush()

    return cloudpickle_file, python_file


def _dump_config_into_tempfile(config: dict) -> IO[str]:
    temp_file = tempfile.NamedTemporaryFile("w")
    temp_file.write(yaml.dump(config))
    temp_file.flush()
    return temp_file


def _launch_serve(
    server_type: str, replica_num: int, env: dict, port: int = 8888
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
            str(find_free_port(port + replica_idx)),
        ]
        processes.append(subprocess.Popen(cmd, env=env))

    return processes


def _launch_data_loader(replica_num: int, env: dict) -> List[subprocess.Popen]:
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


def _launch_nn_worker(nproc_per_node: int, env: dict) -> List[subprocess.Popen]:
    assert 1 <= nproc_per_node <= 8

    processes = []
    for replica_index in range(nproc_per_node):
        current_env = env.copy()
        current_env["WORLD_SIZE"] = str(nproc_per_node)
        current_env["RANK"] = str(replica_index)
        current_env["LOCAL_RANK"] = str(replica_index)

        processes.append(
            subprocess.Popen(
                [
                    "python3",
                    os.environ["PERSIA_NN_WORKER_ENTRY"],
                ],
                env=current_env,
            )
        )
    return processes


def _launch_nats_server(port: int) -> subprocess.Popen:
    return subprocess.Popen(["nats-server", "-p", str(port)])


def _kill_processes(process_group: List[subprocess.Popen]):
    for process in process_group:
        process.kill()

    for process in process_group:
        process.wait()


class PersiaServiceCtx:
    """Launch the required processes to mock the distributed PERSIA environment.

    You can add the embedding config file path or a dict that contains the
    embedding config information, where the embedding configuration will both apply to
    the `embedding_worker` and `embedding_parameter_server`.  Similarly you can also
    set global config using either config file path or a dict.

    For example, training the model with the :class:`.TrainCtx` under the
    :class:`PersiaServiceCtx` context, :class:`PersiaServiceCtx` will launch
    the :class:`.DataCtx` in subprocess.

    .. code-block:: python

        import torch

        from persia.helper import PersiaServiceCtx
        from persia.ctx import DataCtx, TrainCtx
        from persia.data import DataLoader, StreamingDataset
        from persia.embedding import IDTypeFeature, PersiaBatch, Label
        from persia.embedding.optim import Adam

        embedding_config = {"slots_config": {"age": {"dim": 8}}}

        def data_loader_func():
            import numpy as np

            with DataCtx() as data_ctx:
                for i in range(5):
                    persia_batch = PersiaBatch(
                        id_type_features=[
                            IDTypeFeature
                        ],
                        labels=[Label(np.array([0], dtype=np.float32))]
                    )
                    data_ctx.send_data(persia_batch)


        with PersiaServiceCtx(
            embedding_config=embedding_config,
            data_loader_func=data_loader_func
        ):
            prefetch_buffer_size = 15
            with TrainCtx(
                ...
            ) as ctx:
                data_loader = DataLoader(StreamingDataset(prefetch_buffer_size))
                for persia_training_batch in data_loader:
                    ...
    """

    def __init__(
        self,
        data_loader_func: Optional[Callable] = None,
        nn_worker_func: Optional[Callable] = None,
        embedding_config: Optional[dict] = None,
        embedding_config_path: Optional[str] = None,
        global_config: Optional[dict] = None,
        global_config_path: Optional[str] = None,
        data_loader_replica_num: Optional[int] = 1,
        nproc_per_node: Optional[int] = 1,
        embedding_worker_replica_num: Optional[int] = 1,
        embedding_parameter_replica_num: Optional[int] = 1,
        embedding_worker_port: Optional[int] = 7777,
        embedding_parameter_server_port: Optional[int] = 8888,
        nats_server_port: Optional[int] = 4222,
    ):
        """
        Arguments:
            data_loader_func (Callable, optional): data loader function that will be pickled and
                run on the individual process.
            nn_worker_func (Callable, optional): nn_worker function that will be pickled and
                run on the individual process.
            embedding_config (dict, optional): PERSIA embedding config,
                `configuration reference <https://persiaml-tutorials.pages.dev/configuration/#embedding-config>`_.
            embedding_config_path (str, optional): PERSIA embedding config path.
            global_config (dict, optional): PERSIA global config,
                `configuration reference <https://persiaml-tutorials.pages.dev/configuration/#global-configuration>`_.
            global_config_path (str, optional): PERSIA global config path.
            data_loader_replica_num (int, optional): data_loader process number.
            nproc_per_node (int, optional): number of process for data parallel.
            embedding_worker_replica_num (int, optional): number of process for embedding_worker.
            embedding_parameter_replica_num: (int, optional): number of process for
                embedding_parameter_server.
            embedding_worker_port (int, optional): port of embedding-worker server.
            embedding_parameter_server_port (int, optional): port of embedding-parameter-server.
            nats_server_port (int, optional): port of nats-server.
        """

        self.process_group: List[subprocess.Popen] = []
        self.temp_files: List[IO[Union[str, bytes]]] = []

        self.nats_server_port = find_free_port(nats_server_port)
        os.environ["PERSIA_NATS_URL"] = f"localhost:{self.nats_server_port}"
        os.environ["LOG_LEVEL"] = "info"
        self.current_env = os.environ.copy()

        self.data_loader_func = data_loader_func
        self.nn_worker_func = nn_worker_func
        self.data_loader_replica_num = data_loader_replica_num
        self.nproc_per_node = nproc_per_node
        self.embedding_config = embedding_config
        self.global_config = global_config or _DEFAULT_GLOBAL_CONFIG
        self.embedding_config_path = embedding_config_path
        self.global_config_path = global_config_path
        self.embedding_worker_replica_num = embedding_worker_replica_num
        self.embedding_parameter_replica_num = embedding_parameter_replica_num
        self.embedding_worker_port = embedding_worker_port
        self.embedding_parameter_server_port = embedding_parameter_server_port

        self.context_lock = Lock()
        self.thread_worker: Optional[Thread] = None
        self.running: bool = False

    def __enter__(self):
        self.process_group.append(_launch_nats_server(self.nats_server_port))

        if not self.embedding_config_path:
            assert self.embedding_config is not None
            temp_file = _dump_config_into_tempfile(self.embedding_config)
            embedding_config_path = temp_file.name
            self.temp_files.append(temp_file)

        if not self.global_config_path:
            assert self.global_config is not None
            temp_file = _dump_config_into_tempfile(self.global_config)
            global_config_path = temp_file.name
            self.temp_files.append(temp_file)

        self.current_env["PERSIA_EMBEDDING_CONFIG"] = embedding_config_path
        self.current_env["PERSIA_GLOBAL_CONFIG"] = global_config_path

        # Launch embedding-worker
        self.process_group.extend(
            _launch_serve(
                "persia-embedding-worker",
                replica_num=self.embedding_worker_replica_num,
                env=self.current_env,
                port=self.embedding_worker_port,
            )
        )
        # Launch embedding-parameter-server
        self.process_group.extend(
            _launch_serve(
                "persia-embedding-parameter-server",
                replica_num=self.embedding_parameter_replica_num,
                env=self.current_env,
                port=self.embedding_parameter_server_port,
            )
        )

        # Launch dataloader
        if self.data_loader_func is not None:
            temp_files = _dump_function_into_tempfile(self.data_loader_func)
            self.current_env["PERSIA_DATALOADER_ENTRY"] = temp_files[1].name
            self.temp_files.extend(temp_files)
            self.process_group.extend(
                _launch_data_loader(self.data_loader_replica_num, self.current_env)
            )

        # Launch nn-worker
        if self.nn_worker_func is not None:
            temp_files = _dump_function_into_tempfile(self.nn_worker_func)
            self.current_env["PERSIA_NN_WORKER_ENTRY"] = temp_files[1].name
            self.temp_files.extend(temp_files)
            self.process_group.extend(
                _launch_nn_worker(self.nproc_per_node, self.current_env)
            )

        self.running = True

        def _detect_subprocess_terminate(check_interval: int = 2):
            while True:
                with self.context_lock:
                    if self.running:
                        for process in self.process_group:
                            status = process.poll()
                            if status:
                                # Raise Exception if the subprocess is
                                _kill_processes(self.process_group)
                                raise Exception(
                                    f"detect subprocess crash down.., crash down the main thread, \
                                        crash process: {process.args}"
                                )
                    else:
                        break
                time.sleep(check_interval)

        # monitoring the subprocess group status.
        self.thread_worker = Thread(target=_detect_subprocess_terminate, daemon=True)
        self.thread_worker.start()

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        # release resource
        for temp_file in self.temp_files:
            temp_file.close()

        with self.context_lock:
            _kill_processes(self.process_group)
            self.running = False

        if self.thread_worker:
            self.thread_worker.join()


def ensure_persia_service(*args, **kwargs) -> PersiaServiceCtx:
    return PersiaServiceCtx(*args, **kwargs)


if __name__ == "__main__":

    def _data_loader():
        import numpy as np

        from persia.ctx import DataCtx
        from persia.embedding.data import PersiaBatch, IDTypeFeature, Label
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
            labels=[
                Label(
                    np.array(
                        [
                            1,
                        ],
                        dtype=np.float32,
                    )
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
    from persia.data import DataLoader, StreamingDataset
    from persia.embedding import get_default_embedding_config
    from persia.env import get_world_size

    with ensure_persia_service(
        data_loader_func=_data_loader,
        embedding_config=embedding_config,
        global_config=global_config,
        embedding_worker_port=10001,
        embedding_parameter_server_port=10002,
        nats_server_port=10003,
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

            _data_loader = DataLoader(
                StreamingDataset(buffer_size=15), timeout_ms=1000 * 30
            )
            data_generator = iter(_data_loader)
            data = next(data_generator)
