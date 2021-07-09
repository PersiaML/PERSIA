from abc import abstractmethod, ABC
from typing import List, Tuple
import time

from persia.prelude import PyOptimizerBase
from persia.backend import get_backend
from persia.logger import get_default_logger

logger = get_default_logger()


class Optimizer(ABC):
    def __init__(self):
        self.optimizer_base = PyOptimizerBase()

    @abstractmethod
    def apply(self):
        r"""abstraction method for optimizer that support different optimizer register
        function
        """
        ...

    def step(self):
        # TODO: update the sparse grad
        ...

    def register_optimizer(self):
        backend = get_backend()
        backend.register_optimizer(self.optimizer_base)


class SGD(Optimizer):
    r"""A wrapper to config the embedding-server SGD optimizer

    Arguments:
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
        self.optimizer_base.init_sgd(self.lr, self.weight_decay)
        self.register_optimizer()


class Adam(Optimizer):
    r"""A wrapper to config the embedding-server Adam optimizer

    Arguments:
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
        self.optimizer_base.register_adam(
            self.lr, self.betas, self.weight_decay, self.eps
        )


class Adagrad(Optimizer):
    r"""A wrapper to config the embedding-server Adagrad optimizer

    Arguments:
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
        self.optimizer_base.init_adagrad(
            self.lr,
            self.weight_decay,
            self.initial_accumulator_value,
            self.g_square_momentum,
            self.eps,
        )
        self.register_optimizer()
