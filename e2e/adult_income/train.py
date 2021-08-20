import os

import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics

from persia.ctx import TrainCtx, eval_ctx
from persia.sparse.optim import Adagrad
from persia.env import get_rank, get_local_rank, get_world_size
from persia.logger import get_default_logger
from persia.utils import setup_seed
from persia.data import Dataloder, PersiaDataset, StreamingDataset
from persia.prelude import PyPersiaBatchData, PyPersiaBatchDataSender

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


def test(model: torch.nn.Module, data_laoder: Dataloder):
    logger.info("start to test...")
    model.eval()

    with eval_ctx() as ctx:
        accuracies, losses = [], []
        all_pred, all_target = [], []
        for (batch_idx, batch_data) in enumerate(tqdm(data_laoder, desc="test...")):
            dense, sparse, target = ctx.prepare_features(batch_data)
            output = model(dense, sparse)
            loss = loss_fn(output, target)
            all_pred.append(output.cpu().detach().numpy())
            all_target.append(target.cpu().detach().numpy())
            accuracy = (torch.round(output) == target).sum() / target.shape[0]
            accuracies.append(accuracy)
            losses.append(float(loss))

        all_pred, all_target = np.concatenate(all_pred), np.concatenate(all_target)

        fpr, tpr, th = metrics.roc_curve(all_target, all_pred)
        test_auc = metrics.auc(fpr, tpr)

        test_accuracy = torch.mean(torch.tensor(accuracies))
        test_loss = torch.mean(torch.tensor(losses))
        logger.info(
            f"test auc is {test_auc} accuracy is {test_accuracy}, loss is {test_loss}"
        )

    model.train()

    return test_auc, test_accuracy


if __name__ == "__main__":
    model = DNN()
    logger.info("init Simple DNN model...")
    rank, device_id, world_size = get_rank(), get_local_rank(), get_world_size()

    torch.cuda.set_device(device_id)
    model.cuda(device_id)

    dense_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sparse_optimizer = Adagrad(lr=1e-2)
    loss_fn = torch.nn.BCELoss(reduction="mean")
    logger.info("finish genreate dense ctx")

    test_dir = os.path.join("/data/test.npz")
    test_dataset = TestDataset(test_dir, batch_size=128)
    test_interval = 254
    buffer_size = 10

    with TrainCtx(
        sparse_optimizer=sparse_optimizer,
        dense_optimizer=dense_optimizer,
        device_id=device_id,
    ) as ctx:
        train_dataloader = Dataloder(StreamingDataset(buffer_size))
        test_loader = Dataloder(test_dataset, is_training=False)

        for (batch_idx, data) in enumerate(train_dataloader):
            dense, sparse, target = ctx.prepare_features(data)
            output = model(dense, sparse)
            loss = loss_fn(output, target)
            scaled_loss = ctx.backward(loss)
            accuracy = (torch.round(output) == target).sum() / target.shape[0]
            logger.info(
                f"current idx: {batch_idx} loss: {loss} scaled_loss: {scaled_loss} accuracy: {accuracy}"
            )

            if batch_idx % test_interval == 0 and batch_idx != 0:
                test_auc, test_acc = test(model, test_loader)
                assert (
                    test_auc > 0.8
                ), f"test_auc error, expect greater than 0.8 but got {test_auc}"
                break
