import os

from abc import ABC, abstractmethod
from typing import Optional

import torch

from persia.logger import get_default_logger

_logger = get_default_logger()


class DistributedBaseOption(ABC):
    """Implements a common option to convert torch model to a distributed data parallel model,
    e.g. Bagua Distributed or pyTorch DDP.

    This class should not be instantiated directly."""

    def __init__(self, master_port: int, master_addr: Optional[str] = None):
        """
        Arguments:
            master_port (int): master of collective communication ip address.
            master_addr (str, optional): master of collective communication service port.
        """
        self.master_addr = master_addr
        self.master_port = master_port

        if master_addr is not None:
            _logger.info(
                f"Distributed option master addr: {master_addr}, master port: {master_port}"
            )
        else:
            _logger.info(
                f"Distributed option master addr not found, retrieve the master addr by nats service, master port: {master_port}"
            )

    @abstractmethod
    def convert2distributed_model(
        self,
        model: torch.nn.Module,
        world_size: int,
        rank_id: int,
        device_id: Optional[int] = None,
        master_addr: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """

        Arguments:
            model (torch.nn.Module): the PyTorch model that needs to be converted to data-parallel model.
            world_size (int): total number of processes.
            rank_id (int): rank of current process.
            device_id (int, optional): device id for current process.
            master_addr (str, optional): master of the collective communication ip address.
            optimizer (torch.optim.Optimizer, optional): the PyTorch optimizer
                that may need to be converted alongside the model.
        """
        ...

    @abstractmethod
    def init_with_env_file(self) -> bool:
        """Check if the current option was initialized with a ddp env file or not

        Returns:
            ``True`` if the current option was initialized with a ddp env file
        """
        ...


_ddp_backend_support_list = ["nccl", "gloo"]
_ddp_init_method_list = ["tcp", "file"]


class DDPOption(DistributedBaseOption):
    """Implements an option to convert torch model to a DDP model.

    Current backend in :class:`DDPOption` only support `nccl` and `gloo`. You can set
    ``backend="nccl"`` if your PERSIA task is training on the cluster with the CUDA device.
    Or set ``backend="gloo"`` if your PERSIA task is training on the cluster only with the CPU.

    For example:

    .. code-block:: python

        from persia.distributed.DDPOption

        ddp_option = DDPOption(backend="nccl")

    If you want to change the default `master_port` or `master_addr`, add the ``kwargs`` to :class:`DDPOption`.

    .. code-block:: python

        from persia.distributed.DDPOption

        ddp_option = DDPOption(backend="nccl", master_port=23333, master_addr="localhost")
    """

    def __init__(
        self, initialization_method: str = "tcp", backend: str = "nccl", **options
    ):
        """
        Arguments:
            initialization_method (str): the PyTorch distributed initialization_method method,
                support tcp and file currently. See
                `PyTorch initialization <https://pytorch.org/docs/stable/distributed.html#initialization>`_
                for more details.
            backend (str): backend of collective communication. Currently support nccl.
            options (dict): options that include the master_port or master_addr.
        """
        super(DDPOption, self).__init__(
            options.pop("master_port", 23456), options.pop("master_addr", None)
        )

        assert (
            backend in _ddp_backend_support_list
        ), f"The selected backend: {backend} is not support, current backend only \
            support {_ddp_backend_support_list}"
        assert (
            initialization_method in _ddp_init_method_list
        ), f"The selected init_method: {initialization_method} is not support, current init_method \
            only support {_ddp_init_method_list}"

        self.initialization_method = initialization_method
        self.backend = backend
        self.options = options

    def convert2distributed_model(
        self,
        model: torch.nn.Module,
        world_size: int,
        rank_id: int,
        device_id: Optional[int] = None,
        master_addr: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Arguments:
            model (torch.nn.Module): the PyTorch model that needs to be converted to data-parallel model.
            world_size (int): total number of processes.
            rank_id (int): rank of current process.
            device_id (int, optional): device id for current process.
            master_addr (str, optional): master of collective communication ip address.
            optimizer (torch.optim.Optimizer, optional): the PyTorch optimizer
                that may need to be converted alongside the model.
        """

        if self.initialization_method == "tcp":
            assert (
                master_addr or self.master_addr
            ) and self.master_port, "Master IP and Port empty, pytorch DDP should pass master addr and\
            master port!"

            master_addr = self.master_addr or master_addr
            init_method = (
                f"{self.initialization_method}://{master_addr}:{self.master_port}"
            )
        elif self.initialization_method == "file":
            sync_file = self.options.pop("sync_file", None)
            assert (
                sync_file
            ), "Launch pytorch ddp with file init_method, should pass the sync_file filepath as argument\
                for constructor function"

            if os.path.exists(sync_file):
                raise Exception(
                    f"Pytorch ddp sync file already exists {sync_file}, delete the sync file before launch\
                    the persia task!"
                )

            init_method = f"{self.initialization_method}://{sync_file}"
        else:
            raise NotImplementedError

        torch.distributed.init_process_group(
            self.backend,
            init_method=init_method,
            rank=rank_id,
            world_size=world_size,
        )
        _logger.info(
            f"Pytorch ddp init process group done, corresponding backend is {self.backend}, init\
            method is {self.initialization_method}"
        )

        device_ids = [device_id] if self.backend != "gloo" else None
        parallel_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=device_id,
            find_unused_parameters=True,
        )

        return parallel_model, optimizer

    def init_with_env_file(self) -> bool:
        """Check if the current option was initialized with a ddp env file or not

        Returns:
            ``True`` if the current option was initialized with a ddp env file
        """
        return self.initialization_method == "file"


def _select_bagua_algorithm(
    algorithm: str = "gradient_allreduce",
    model: Optional[torch.nn.Module] = None,
    **options,
):
    """Select corresponding bagua algorithm for current training.

    Arguments:
        algorithm (str): name of Bagua algorithm.
        model (torch.nn.Model): the pytorch model that need to converted to dataparallel model which is
            needed when apply bagua QAdam algorithm.
        options (dict): options for bagua algorithm.

    Returns:
        Bagua distributed training algorithm and corresponding optimizer.
    """
    optimizer = None

    # pytype: disable=import-error
    if algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.DecentralizedAlgorithm()
    elif algorithm == "low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
    elif algorithm == "qadam":
        assert (
            optimizer is not None
        ), "Bagua qadam algorithm need to pass optimizer as argument, but got None."
        from bagua.torch_api.algorithms import q_adam

        optimizer = q_adam.QAdamOptimizer(model.parameters(), **options)
        algorithm = q_adam.QAdamAlgorithm(optimizer)
    elif algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.ByteGradAlgorithm()
    elif algorithm == "async":
        from bagua.torch_api.algorithms import async_model_average

        algorithm = async_model_average.AsyncModelAverageAlgorithm(**options)
    else:
        _logger.error(f"Bagua algorithm not found, {algorithm} not implement.")
        raise NotImplementedError
    # pytype: enable=import-error

    return algorithm, optimizer


class BaguaDistributedOption(DistributedBaseOption):
    """Implements an option to convert torch model to a bagua distributed model.

    Example for :class:`BaguaDistributedOption`:

    .. code-block:: python

        from persia.distributed import BaguaDistributedOption

        kwargs = {
            "enable_bagua_net": True
        }
        bagua_option = BaguaDistributedOption("gradient_allreduce", **kwargs)

    Algorithms supported in `Bagua`:

    +-----------------------------+
    |Algorithm Name               |
    +=============================+
    |gradient_allreduce           |
    +-----------------------------+
    |decentralized                |
    +-----------------------------+
    |low_precision_decentralized  |
    +-----------------------------+
    |qadam                        |
    +-----------------------------+
    |bytegrad                     |
    +-----------------------------+
    |async                        |
    +-----------------------------+

    .. note::
        You can review `Bagua Algorithm <https://tutorials.baguasys.com/algorithms/>`_ for more details,
        especially the arguments of algorithm.

    .. note::
        The :class:`BaguaDistributedOption` only supports the `CUDA` environment, if you want to run PERSIA task
        on the CPU cluster, try :class:`DDPOption` with `backend='gloo'` instead of
        :class:`BaguaDistributedOption`.
    """

    def __init__(self, algorithm: str, **options):
        """
        Arguments:
            algorithm (str): name of Bagua algorithm.
            options (dict): options for Bagua algorithm
        """
        super(BaguaDistributedOption, self).__init__(
            options.pop("master_port", 23456), options.pop("master_addr", None)
        )

        assert (
            torch.cuda.is_available()
        ), "BaguaDistributedOption only support on cuda device."
        self.algorithm = algorithm
        self.options = options

    def convert2distributed_model(
        self,
        model: torch.nn.Module,
        world_size: int,
        rank_id: int,
        device_id: Optional[int] = None,
        master_addr: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Arguments:
            model (torch.nn.Module): the PyTorch model that needs to be converted to data-parallel model.
            world_size (int): total number of processes.
            rank_id (int): rank of current process.
            device_id (int, optional): device id for current process.
            master_addr (str, optional): master of collective communication ip address.
            optimizer (torch.optim.Optimizer, optional): the PyTorch optimizer
                that may need to be converted alongside the model.
        """

        try:
            # pytype: disable=import-error
            from bagua.torch_api import (
                init_process_group,
            )

            # pytype: enable=import-error
        except ImportError as e:
            _logger.error("Import Bagua Error, install Bagua before use BaguaOption")
            raise e

        current_env = os.environ
        current_env["MASTER_ADDR"] = self.master_addr or master_addr
        current_env["MASTER_PORT"] = str(self.master_port)
        current_env["BAGUA_DEFAULT_BUCKET_SIZE"] = str(
            self.options.pop("default_bucket_size", 10 * 1024**2)
        )

        # autotune option
        autotune_level = int(self.options.pop("autotune_level", "0"))
        autotune_service_port = int(self.options.pop("bagua_service_port", "29501"))

        current_env["BAGUA_SERVICE_PORT"] = str(autotune_service_port)
        current_env["BAGUA_AUTOTUNE"] = str(autotune_level)
        current_env["BAGUA_AUTOTUNE_MAX_SAMPLES"] = str(
            self.options.pop("autotune_max_samples", 60)
        )
        current_env["BAGUA_AUTOTUNE_WARMUP_TIME_S"] = str(
            self.options.pop("autotune_warmup_time", 30)
        )
        current_env["BAGUA_IS_OUTPUT_AUTOTUNE_LOG"] = str(
            int(self.options.pop("is_output_autotune_log", False))
        )
        current_env["BAGUA_AUTOTUNE_SAMPLING_CONFIDENCE_TIME_S"] = str(
            self.options.pop("autotune_sampling_confidence_time", 5.0)
        )
        if autotune_level > 0:
            current_env[
                "AUTO_TUNE_SERVER_ADDR"
            ] = f"{master_addr}:{autotune_service_port}"

        enable_bagua_net = self.options.pop("enable_bagua_net", False)
        if enable_bagua_net:
            import pkg_resources

            current_env["LD_LIBRARY_PATH"] = "{}:{}".format(
                pkg_resources.resource_filename("bagua_core", "./data/bagua-net"),
                current_env["LD_LIBRARY_PATH"],
            )
            _logger.info("Enable bagua net.")

        init_process_group()
        _logger.info("Bagua init process group done")

        algorithm, algorithm_optimizer = _select_bagua_algorithm(
            self.algorithm, model=model, **self.options
        )

        if algorithm_optimizer is not None:
            _logger.info(
                f"Use converted optimizer! {optimizer} => {algorithm_optimizer}"
            )

            optimizer = algorithm_optimizer

        return model.with_bagua([optimizer], algorithm), optimizer

    def init_with_env_file(self) -> bool:
        """Check if the current option is initiad with a ddp environment file.

        Returns:
            Whether distributed option init with env file.
        """
        return False


def get_default_distributed_option(device_id: Optional[int] = None) -> DDPOption:
    """Get default distributed option.

    Arguments:
        device_id (int, optional): CUDA device_id. Apply ``backend="nccl"`` to the ``DDPOption``
            if the `device_id` not None, otherwise use the ``backend="gloo"`` for CPU only mode.

    Returns:
        Default distributed option.
    """
    if device_id is not None:
        backend = "nccl"
    else:
        backend = "gloo"

    return DDPOption(backend=backend)
