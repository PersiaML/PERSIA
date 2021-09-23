import os

from abc import ABC, abstractmethod
from typing import Optional

import torch

from persia.logger import get_default_logger

_logger = get_default_logger()


class DistributedBaseOption(ABC):
    def __init__(self, master_port: int, master_addr: str = None) -> None:
        self.master_addr = master_addr
        self.master_port = master_port

    @abstractmethod
    def convert2distributed_model(
        self,
        model: torch.nn.Module,
        world_size: int,
        local_world_size: int,
        rank_id: int,
        device_id: Optional[int] = None,
        master_addr: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        ...

    @abstractmethod
    def init_with_env_file(self) -> bool:
        ...

_ddp_backend_support_list = ["nccl"]
_ddp_init_method_list = ["tcp", "file"]


class DDPOption(DistributedBaseOption):
    def __init__(self, init_method: str = "tcp", backend: str = "nccl", **options) -> None:
        super(DDPOption, self).__init__(options.pop("master_port", 23456), options.pop("master_addr", None))

        assert (
            backend in _ddp_backend_support_list
        ), f"The selected backend: {backend} is not support, current backend only support {_ddp_backend_support_list}"
        assert (
            init_method in _ddp_init_method_list
        ), f"The selected init_method: {init_method} is not support, current init_method only support {_ddp_init_method_list}"

        self.init_method = init_method
        self.backend = backend
        self.options = options

    def convert2distributed_model(
        self,
        model: torch.nn.Module,
        world_size: int,
        rank_id: int,
        device_id: Optional[int] = None,
        master_addr: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        
        if self.init_method == "tcp":
            assert (
                master_addr and self.master_port
            ), "Master IP and Port empty, pytorch DDP should pass master addr and master port!"
            master_addr = self.master_addr or master_addr
            init_method = f"{self.init_method}://{master_addr}:{self.master_port}"
        elif self.init_method == "file":
            sync_file = self.options.pop("sync_file", None)
            assert sync_file, "Launch pytorch ddp with init_method file, should pass the sync_file filepath for DistributedOption"
            if os.path.exists(sync_file):
                _logger.warn("Delete the sync file before launch the persia task!")  

                raise Exception(f"Pytorch ddp sync file already exists {sync_file}")

            init_method = f"{self.init_method}://{sync_file}"
        else:
            raise NotImplementedError
        
        torch.distributed.init_process_group(
            self.backend,
            init_method=init_method,
            rank=rank_id,
            world_size=world_size,
        )
        _logger.info("Pytorch ddp init process group done")

        parallel_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=True,
        )

        return parallel_model, optimizer

    def init_with_env_file(self) -> bool:
        return self.init_method == "file"

def _select_bagua_algorithm(
    algorithm: str = "gradient_allreduce",
    model: Optional[torch.nn.Module] = None,
    **options,
):
    optimizer = None

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

    return algorithm, optimizer


class BaguaDistributedOption(DistributedBaseOption):
    def __init__(self, algorithm: str, **options) -> None:
        super(BaguaDistributedOption, self).__init__(options.pop("master_port", 23456), options.pop("master_addr", None))

        self.algorithm = algorithm
        self.options = options

    def convert2distributed_model(
        self,
        model: torch.nn.Module,
        _world_size: int,
        _rank_id: int,
        _device_id: Optional[int] = None,
        master_addr: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        try:
            from bagua.torch_api import init_process_group
        except ImportError as e:
            _logger.error(
                "Import Bagua Error, instasll Bagua before use BaguaOption, you should build the persia docker image with Bagua before task start."
            )
            raise e

        current_env = os.environ
        # current_env["WORLD_SIZE"] = str(world_size)
        # current_env["LOCAL_WORLD_SIZE"] = str(local_world_size)
        # current_env["RANK"] = str(rank_id)
        # current_env["LOCAL_RANK"] = str(device_id)
        current_env["MASTER_ADDR"] = self.master_addr or master_addr
        current_env["MASTER_PORT"] = str(self.master_port)
        current_env["BAGUA_DEFAULT_BUCKET_SIZE"] = str(self.options.pop("default_bucket_size", 10 * 1024 ** 2))

        # autotune option 
        autotune_level = int(self.options.pop("autotune_level", "0"))
        autotune_service_port = int(self.options.pop("bagua_service_port", "29501"))

        current_env["BAGUA_SERVICE_PORT"] = str(autotune_service_port)
        current_env["BAGUA_AUTOTUNE"] = str(autotune_level)
        current_env["BAGUA_AUTOTUNE_MAX_SAMPLES"] = str(self.options.pop("autotune_max_samples", 60))
        current_env["BAGUA_AUTOTUNE_WARMUP_TIME_S"] = str(self.options.pop("autotune_warmup_time", 30))
        current_env["BAGUA_IS_OUTPUT_AUTOTUNE_LOG"] = str(int(self.options.pop("is_output_autotune_log", False)))
        current_env["BAGUA_AUTOTUNE_SAMPLING_CONFIDENCE_TIME_S"] = str(
            self.options.pop("autotune_sampling_confidence_time", 5.0)
        )
        if autotune_level > 0:
            current_env["AUTO_TUNE_SERVER_ADDR"] = f"{master_addr}:{autotune_service_port}"

        enable_bagua_net = self.options.pop("enable_bagua_net", False)
        if enable_bagua_net:
            import pkg_resources

            current_env["LD_LIBRARY_PATH"] = "{}:{}".format(
                pkg_resources.resource_filename("bagua_core", "./data/bagua-net"),
                current_env["LD_LIBRARY_PATH"],
            )

        init_process_group()
        _logger.info("Bagua init process group done")

        algorithm, algorithm_optimizer  = _select_bagua_algorithm(
            self.algorithm, model=model, **self.options
        )

        if algorithm_optimizer is not None:
            _logger.info(f"Use converted optimizer! {optimizer} => {algorithm_optimizer}")

            optimizer = algorithm_optimizer
            

        return model.with_bagua([optimizer], algorithm), optimizer

    def init_with_env_file(self) -> bool:
        return False

def get_default_distributed_option() -> DDPOption:
    return DDPOption()
