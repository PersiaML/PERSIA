import os
import torch


_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
_REPLICA_SIZE = int(os.environ.get("REPLICA_SIZE", 1))
_REPLICA_INDEX = int(os.environ.get("REPLICA_INDEX", 0))

if torch.cuda.is_available():
    _RANK_ID = int(os.environ.get("RANK", 0))
    _LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
else:
    _RANK_ID = -1
    _LOCAL_RANK = -1


def get_world_size() -> int:
    """Get the total number of processes."""
    return _WORLD_SIZE


def get_rank() -> int:
    """Get the rank of this process."""
    return _RANK_ID


def get_local_rank() -> int:
    """Get the local rank of this process (the rank of the process on the local machine)."""
    return _LOCAL_RANK


def get_replica_size() -> int:
    """Get the replica size of current service.The replica size is the scale number of
    service which launch by docker service or k8s"""
    return _REPLICA_SIZE


def get_replica_index() -> int:
    """Get the replica index of current service.The replica index is the launch order of current service"""

    return _REPLICA_INDEX
