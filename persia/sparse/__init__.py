from uuid import uuid4
from typing import List, Tuple

import torch

__EMB_ID__ = set()
# all embeddings share one piece cuda device memory
__EMBEDDING_FUSION__ = False
__HALF_TRAINING__ = True


def set_embedding_fusion(flag: bool):
    global __name__
    __EMBEDDING_FUSION__ = flag


def set_half(flag: bool):
    global __HALF_TRAINING__
    __HALF_TRAINING__ = flag


def gen_emb_id() -> str:
    return str(uuid4())
