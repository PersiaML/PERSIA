from abc import abstractmethod, ABC
from typing import List, Tuple

from persia.prelude import PyOptimizerBase


class Optimizer(ABC):
    def __init__(self):
        self.optimizer_base = PyOptimizerBase()

    @abstractmethod
    def apply(self):
        # TODO:
        # update the optimzier args directly
        ...

    def step(self):
        # TODO: update the sparse grad
        ...


class SGD(Optimizer):
    r"""A wrapper to config the embedding-server SGD optimizer
    Args:
        params(float): learning rate
        momentum(float, optional): momentum factor
        weight_decay(float, optional): parameters L2 penalty factor
    """

    def __init__(self, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def apply(self):
        self.optimizer_base.register_sgd(self.lr, self.weight_decay)


class Adam(Optimizer):
    r"""A wrapper to config the embedding-server Adam optimizer
    Args:
        lr(float): learning rate
        betas(tuple[float,float], optional): caculate the running averages of gradient and its square
        weight_decay(float, optional): parameters L2 penalty factor
        eps(float, optional): epsilon to avoid div zero
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        eps: float = 1e-8,
    ):
        super(Adam, self).__init__()
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def apply(self):
        ...


class Adagrad(Optimizer):
    r"""A wrapper to config the embedding-server Adagrad optimizer
    Args:
        lr(float): learning rate
        initial_accumulator_value(float, optional): initialization accumulator value for adagrad optimzer
        g_square_momentum(float, optional): factor of accumulator incremental
        eps(float): epsilon to avoid div zero
    """

    def __init__(
        self,
        lr: float = 1e-2,
        initial_accumulator_value: float = 1e-2,
        weight_decay: float = 0,
        g_square_momentum: float = 1,
        eps: float = 1e-10,
    ):
        super(Adagrad, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.g_square_momentum = g_square_momentum
        self.eps = eps

    def apply(self):
        self.optimizer_base.register_adagrad(
            self.lr,
            self.weight_decay,
            self.initial_accumulator_value,
            self.g_square_momentum,
            self.eps,
        )
