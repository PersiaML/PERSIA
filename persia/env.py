import os

try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    # fallback implementation of cache_property for Python < 3.8
    def cached_property(func):
        return property(lru_cache()(func))


class ENV:
    REPLICA_SIZE = None
    REPLICA_INDEX = None
    WORLD_SIZE = None
    RANK_ID = None
    LOCAL_RANK = None

    ENV_LAUNCHER = None

    def env_check(self) -> bool:
        """Check current environment is valid or not."""

        rank_env = (
            self.WORLD_SIZE is not None
            and self.RANK_ID is not None
            and self.LOCAL_RANK is not None
        )
        replica_env = self.REPLICA_INDEX is not None and self.REPLICA_SIZE is not None

        if not (rank_env or replica_env):
            raise Exception(
                f"""PersiaEnv check failed, aleast pass rank or replica info to run the persia task,
                    launcher: {self.ENV_LAUNCHER} world_size: {self.WORLD_SIZE},
                    rank_id: {self.RANK_ID}, local_rank: {self.LOCAL_RANK}, replica_size: {self.REPLICA_SIZE},
                    replica_index: {self.REPLICA_INDEX}"""
            )

    @cached_property
    def rank(self):
        # TODO: Check the environment and cache the value.
        return self.RANK_ID

    @cached_property
    def local_rank(self):
        return self.LOCAL_RANK

    @cached_property
    def replica_index(self):
        return self.REPLICA_INDEX

    @cached_property
    def replica_size(self):
        return self.REPLICA_SIZE

    @cached_property
    def world_size(self):
        return self.WORLD_SIZE


class HonchoENV(ENV):

    ENV_LAUNCHER = "honcho"

    def __init__(self):
        """Honcho environment have the basic ability to support local training, typically used at single machine training."""
        honcho_process_name = os.environ["HONCHO_PROCESS_NAME"]

        if os.environ.get("RANK", None):
            self.RANK_ID = int(os.environ["RANK"])
            self.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
            self.WORLD_SIZE = int(os.environ.get("WORLD_SIZE"))
        else:
            _, process_idx = honcho_process_name.split(".")
            process_idx = int(process_idx) - 1

            self.REPLICA_INDEX = int(os.environ.get("REPLICA_INDEX"))
            self.REPLICA_SIZE = int(os.environ.get("REPLICA_SIZE"))


class DockerENV(ENV):

    ENV_LAUNCHER = "docker"

    def __init__(self):
        """Docker environment that launch the PersiaML task by docker-compose."""
        if os.environ.get("RANK", None):
            self.RANK_ID = int(os.environ["RANK"])
            self.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
            self.WORLD_SIZE = int(os.environ.get("WORLD_SIZE"))
        else:
            self.REPLICA_INDEX = int(os.environ.get("TASK_SLOT_ID")) - 1
            self.REPLICA_SIZE = int(os.environ.get("REPLICAS"))


class DefaultENV(ENV):

    ENV_LAUNCHER = "default"

    def __init__(self):
        """Default environment that receive the environment by user manually expose to env."""
        self.WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))
        self.REPLICA_SIZE = int(os.environ.get("REPLICA_SIZE", -1))

        self.REPLICA_INDEX = int(os.environ.get("REPLICA_INDEX", -1))
        self.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
        self.RANK_ID = int(os.environ.get("RANK", -1))


def parse_env() -> ENV:
    """Parse the environment by specific environment keyword"""
    if os.environ.get("HONCHO", None):
        env = HonchoENV()
    elif os.environ.get("DOCKER_COMPOSE", None):
        env = DockerENV()
    else:
        env = DefaultENV()

    env.env_check()
    return env


_env = parse_env()


def get_world_size() -> int:
    """Get the total number of processes."""
    return _env.WORLD_SIZE


def get_rank() -> int:
    """Get the rank of current process."""
    return _env.LOCAL_RANK


def get_local_rank() -> int:
    """Get the local rank of current process.

    Local rank is the rank of the process on the local machine."""
    return _env.LOCAL_RANK


def get_replica_size() -> int:
    """Get the replica size of the current service.

    Replica size is the number of services launched by docker service or k8s"""
    return _env.REPLICA_SIZE


def get_replica_index() -> int:
    """Get the replica index of current service.

    The replica index is a unique identifier assigned to each replica. They are assigned following the order of launching.
    """
    return _env.REPLICA_INDEX
