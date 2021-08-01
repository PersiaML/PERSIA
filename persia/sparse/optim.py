from abc import abstractmethod, ABC
from typing import Tuple

from persia.prelude import PyOptimizerBase
from persia.backend import get_backend


class Optimizer(ABC):
    r"""Base optimizer to configurate the sparse embedding update behavior."""

    def __init__(self):
        self.optimizer_base = PyOptimizerBase()

    @abstractmethod
    def apply(self):
        r"""Abstraction method for optimizer that support register different type of optimizer."""
        ...

    def register_optimizer(self):
        """Register sparse optimizer to embedding server."""
        get_backend().register_optimizer(self.optimizer_base)


class SGD(Optimizer):
    r"""A wrapper to config the embedding-server SGD optimizer."""

    def __init__(self, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Arguments:
            params(float): learning rate.
            momentum(float, optional): momentum factor.
            weight_decay(float, optional): parameters L2 penalty factor.
        """
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def apply(self):
        """Initialize optimizer and register to embedding server."""
        self.optimizer_base.init_sgd(self.lr, self.weight_decay)
        self.register_optimizer()


class Adam(Optimizer):
    r"""A wrapper to config the embedding-server Adam optimizer."""

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        eps: float = 1e-8,
    ):
        """
        Arguments:
            lr(float): learning rate.
            betas(tuple[float,float], optional): caculate the running averages of gradient and its square.
            weight_decay(float, optional): parameters L2 penalty factor.
            eps(float, optional): epsilon to avoid div zero.
        """
        super(Adam, self).__init__()
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def apply(self):
        """Initialize Adam optimizer and register to embedding server."""
        self.optimizer_base.init_adam(self.lr, self.betas, self.eps)
        self.register_optimizer()


class Adagrad(Optimizer):
    r"""A wrapper to config the embedding-server Adagrad optimizer."""

    def __init__(
        self,
        lr: float = 1e-2,
        initial_accumulator_value: float = 1e-2,
        weight_decay: float = 0,
        g_square_momentum: float = 1,
        eps: float = 1e-10,
    ):
        """
        Arguments:
            lr (float): learning rate.
            initial_accumulator_value (float, optional): initialization accumulator value for adagrad optimizer.
            weight_decay (float, optional): parameters L2 penalty factor.
            g_square_momentum (float, optional): factor of accumulator incremental.
            eps(float, optional): epsilon term to avoid divide zero.

        """
        super(Adagrad, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.g_square_momentum = g_square_momentum
        self.eps = eps

    def apply(self):
        """Initialize Adagrad optimizer and register to embedding server."""
        self.optimizer_base.init_adagrad(
            self.lr,
            self.weight_decay,
            self.initial_accumulator_value,
            self.g_square_momentum,
            self.eps,
        )
        self.register_optimizer()
