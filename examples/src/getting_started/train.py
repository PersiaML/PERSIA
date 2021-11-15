import os
from posixpath import abspath

import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics

from persia.ctx import TrainCtx, eval_ctx, EmbeddingConfig
from persia.sparse.optim import Adagrad
from persia.env import get_rank, get_local_rank, get_world_size
from persia.logger import get_default_logger
from persia.data import Dataloder, PersiaDataset, StreamingDataset
from persia.prelude import PyPersiaBatchData, PyPersiaBatchDataSender
from persia.utils import setup_seed

from model import DNN
from data_generator import make_dataloader


logger = get_default_logger("trainer")

device_id = get_local_rank()
setup_seed(3)


class TestDataset(PersiaDataset):
    def __init__(self, test_dir: str, batch_size: int = 128):
        super(TestDataset, self).__init__(buffer_size=10)
        size, loader = make_dataloader(test_dir, batch_size)
        self.loader = loader
        self.loader_size = size

        logger.info(f"test dataset size is {size}")

    def fetch_data(self, persia_sender_channel: PyPersiaBatchDataSender):
        logger.info("test loader start to generate data...")
        for idx, (dense, batch_sparse_ids, target) in enumerate(
            tqdm(self.loader, desc="gen batch data")
        ):
            batch_data = PyPersiaBatchData()
            batch_data.add_dense([dense])
            batch_data.add_sparse(batch_sparse_ids)
            batch_data.add_target(target)
            persia_sender_channel.send(batch_data)

    def __len__(self):
        return self.loader_size


class TrainDataset(PersiaDataset):
    def __init__(self, train_dir: str, batch_size: int = 128):
        super(TestDataset, self).__init__(buffer_size=10)
        size, loader = make_dataloader(test_dir, batch_size)
        self.loader = loader
        self.loader_size = size

        logger.info(f"test dataset size is {size}")

    def fetch_data(self, persia_sender_channel: PyPersiaBatchDataSender):
        logger.info("test loader start to generate data...")
        for idx, (dense, batch_sparse_ids, target) in enumerate(
            tqdm(self.loader, desc="gen batch data")
        ):
            batch_data = PyPersiaBatchData()
            batch_data.add_dense([dense])
            batch_data.add_sparse(batch_sparse_ids)
            batch_data.add_target(target)
            persia_sender_channel.send(batch_data)

    def __len__(self):
        return self.loader_size


def test(model: torch.nn.Module, data_laoder: Dataloder, cuda: bool = True):
    logger.info("start to test...")
    model.eval()

    with eval_ctx(model=model) as ctx:
        accuracies, losses = [], []
        all_pred, all_target = [], []
        for (batch_idx, batch_data) in enumerate(tqdm(data_laoder, desc="test...")):
            (pred, target) = ctx.forward(batch_data)
            loss = loss_fn(pred, target)
            if cuda:
                pred = pred.cpu()
                target = target.cpu()
            else:
                target = target.clone()  # cpu mode need copy the target data..
            all_pred.append(pred.detach().numpy())
            all_target.append(target.detach().numpy())
            accuracy = (torch.round(pred) == target).sum() / target.shape[0]
            accuracies.append(accuracy)
            losses.append(loss)

        all_pred, all_target = np.concatenate(all_pred), np.concatenate(all_target)

        fpr, tpr, th = metrics.roc_curve(all_target, all_pred)
        test_auc = metrics.auc(fpr, tpr)

        test_accuracies = torch.mean(torch.tensor(accuracies))
        test_loss = torch.mean(torch.tensor(losses))
        logger.info(
            f"test auc is {test_auc} accuracy is {test_accuracies}, loss is {test_loss}"
        )

    model.train()


if __name__ == "__main__":
    model = DNN()
    logger.info("init Simple DNN model...")
    rank, device_id, world_size = get_rank(), get_local_rank(), get_world_size()

    cuda = bool(int(os.environ.get("ENABLE_CUDA",1)))
    mixed_precision = True

    if cuda:
        torch.cuda.set_device(device_id)
        model.cuda(device_id)
    else:
        mixed_precision = False
        device_id = None
    dense_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sparse_optimizer = Adagrad(lr=1e-2)
    loss_fn = torch.nn.BCELoss(reduction="mean")
    logger.info("finish genreate dense ctx")

    test_dir = os.path.abspath(os.path.join(__file__, "../data_source/test.npz"))
    test_dataset = TestDataset(test_dir, batch_size=128)
    test_interval = 254
    buffer_size = 10

    embedding_config = EmbeddingConfig()

    with TrainCtx(
        model=model,
        sparse_optimizer=sparse_optimizer,
        dense_optimizer=dense_optimizer,
        mixed_precision=mixed_precision,
        device_id=device_id,
        embedding_config=embedding_config,
    ) as ctx:
        train_dataloader = Dataloder(StreamingDataset(buffer_size))
        test_loader = Dataloder(test_dataset, is_training=False)

        logger.info("start to training...")
        for (batch_idx, data) in enumerate(train_dataloader):
            (output, target) = ctx.forward(data)
            loss = loss_fn(output, target)
            scaled_loss = ctx.backward(loss)
            accuracy = (torch.round(output) == target).sum() / target.shape[0]
            logger.info(
                f"current idx: {batch_idx} loss: {float(loss)} scaled_loss: {float(scaled_loss)} accuracy: {float(accuracy)}"
            )

            if batch_idx % test_interval == 0 and batch_idx != 0:
                test(model, test_loader, cuda)
                break
