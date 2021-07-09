import os
import torch

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DISTRIBUTED = WORLD_SIZE > 1

if torch.cuda.is_available():
    RANK_ID = int(os.environ.get("RANK", 0))
    DEVICE_ID = int(os.environ.get("LOCAL_RANK", 0))
    IS_MASTER = RANK_ID == 0
else:
    DEVICE_ID = -1
    RANK_ID = -1
    IS_MASTER = -1

LOCAL_RANK = DEVICE_ID
REPLICA_SIZE = int(os.environ.get("REPLICA_SIZE", 1))
REPLICA_INDEX = int(os.environ.get("REPLICA_INDEX", 0))
