import torch

from persia.prelude import PyForward


class Dataloder:
    def __init__(self, input_stream, timeout: int):
        self.input_stream = input_stream
        # TODO: box the dataloader to make it clear
        # from persia_embedding_py_client_sharded_server import TrainDataloder
        # self._persia_loader = TrainDataloder(input_stream)
        # FiniteStream: provide a dataset to generate batch data
        # InfiniteStream: provide a stream to load(usally stop by timeout controller)
        self.timeout = timeout

    def __next__(self):
        # TODO: warp the rust exception to python Exception
        while not self.input_stream.is_end():
            yield self.input_stream.read(self.timeout)


class InfiniteIterator(torch.utils.data.IterableDataset):
    def __init__(
        self, forward_engine: PyForward, rectify_factor: float, timeout: int
    ):
        self.timeout = timeout
        self.forward_engine = forward_engine
        self.rectify_factor = rectify_factor

    def __iter__(self):
        self.forward_engine.launch(
           torch.cuda.current_device(), self.rectify_factor,
        )

        while True:
            yield self.forward_engine.get_batch(self.timeout)
