from typing import List, Tuple

import torch


class Embedding(torch.autograd.Function):
    def __init__(
        self, emb_name: str, emb_id: str, emb_dim: int, emb_size: int, initialization
    ):
        self.emb_name = emb_name
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.emb_id = emb_id  # some emb_id share the feature space
        self.initialization = initialization

    def register(self):
        # TODO: implement register function for py_client
        # from persia.prelude import register_embedding
        ...


class VarLenEmbedding(Embedding):
    def __init__(
        self,
        emb_name: str,
        emb_id: str,
        emb_dim: int,
        emb_size: int,
        initialization,
        sample_fixed_size: int,
    ):
        super(VarLenEmbedding, self).__init__(
            emb_name, emb_id, emb_dim, emb_size, initialization
        )

    @staticmethod
    def forward(self):
        ...

    @staticmethod
    def backward(self):
        ...


class SumEmbedding(Embedding):
    def __init__(
        self,
        emb_name: str,
        emb_id: str,
        emb_dim: int,
        emb_size: int,
        initialization,
    ):
        super(SumEmbedding, self).__init__(
            emb_name, emb_id, emb_dim, emb_size, initialization
        )

    @staticmethod
    def forward(self, tensor):
        ...

    @staticmethod
    def backward(self, grad_output):
        ...


class EmbeddingFunction(torch.autograd.Function):
    def __init__(self, embedding_specs: List[Embedding]):
        self.embedding_specs = embedding_specs

    @staticmethod
    def forward(ctx, sparses_raw: List):
        ...

    @staticmethod
    def backward(self, grad_output):
        ...
