from typing import Tuple


class EmbeddingConfig:
    r"""Embedding hyperparameters, argument of :class:`.EmbeddingCtx`."""

    def __init__(
        self,
        emb_initialization: Tuple[float, float] = (-0.01, 0.01),
        admit_probability: float = 1.0,
        weight_bound: float = 10,
    ):
        """
        Arguments:
            emb_initialization (Tuple[float, float], optional): lower and upper bound of embedding uniform initialization.
            admit_probability (float, optional): the probability (0<=, <=1) of admitting a new embedding.
            weight_bound (float, optional): restrict each element value of an embedding in [-weight_bound, weight_bound].
        """
        self.emb_initialization = emb_initialization
        self.admit_probability = admit_probability
        self.weight_bound = weight_bound


def get_default_embedding_config():
    """Get default embedding configuration."""
    return EmbeddingConfig()
