import os

from persia.env import (
    get_local_rank,
    get_rank,
    get_world_size,
    get_replica_index,
    get_replica_size,
    reload_env,
)


def test_nn_worker_env():
    rank = 0
    local_rank = 0
    world_size = 1
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    reload_env()
    assert get_world_size() == world_size
    assert get_rank() == rank
    assert get_local_rank() == local_rank


def test_data_loader_env():
    replica_index = 0
    replica_size = 1

    del os.environ["RANK"]
    os.environ["REPLICA_SIZE"] = str(replica_size)
    os.environ["REPLICA_INDEX"] = str(replica_index)

    reload_env()
    assert get_replica_size() == replica_size
    assert get_replica_index() == replica_index
