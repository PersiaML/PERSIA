import torch


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
    r"""InfiniteIterator for streaming data stop by timeout exception

    Arguments:
        forward_engine : rust forward engine wrapper for fetch input data
        port (int): port for input server to bind
        data_queue_size (int): buffer size for data forward phase
        timeout (int): timeout for data fetch
    """

    def __init__(
        self,
        forward_engine,
        disorder_tolerance: float,
        timeout: int,
        num_forward_workers: int,
    ):
        self.timeout = timeout
        self.forward_engine = forward_engine
        self.disorder_tolerance = disorder_tolerance
        self.num_forward_workers = num_forward_workers

    def __iter__(self):
        self.forward_engine.launch(
            torch.cuda.current_device(),
            self.disorder_tolerance,
            self.num_forward_workers,
        )

        while True:
            yield self.forward_engine.get_batch(self.timeout)
