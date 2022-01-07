import os

import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics

from persia.ctx import TrainCtx, eval_ctx
from persia.embedding.optim import Adagrad
from persia.embedding.data import PersiaBatch
from persia.env import get_rank, get_local_rank, get_world_size
from persia.logger import get_default_logger
from persia.data import DataLoader, IterableDataset, StreamingDataset
from persia.utils import setup_seed

from model import DNN
from data_generator import make_dataloader


logger = get_default_logger("nn_worker")

CPU_TEST_AUC = 0.8928645493226243
GPU_TEST_AUC = 0.8927145127622554


class TestDataset(IterableDataset):
    def __init__(self, test_dir: str, batch_size: int = 128):
        super(TestDataset, self).__init__(buffer_size=10)
        self.loader = make_dataloader(test_dir, batch_size)

    def __iter__(self):
        logger.info("test loader start to generating data...")
        for non_id_type_feature, id_type_features, label in self.loader:
            yield PersiaBatch(
                id_type_features,
                non_id_type_features=[non_id_type_feature],
                labels=[label],
                requires_grad=False,
            )


def test(model: torch.nn.Module, data_loader: DataLoader, cuda: bool):
    logger.info("start to test...")
    model.eval()

    with eval_ctx(model=model) as ctx:
        accuracies, losses = [], []
        all_pred, all_labels = [], []
        for (batch_idx, batch_data) in enumerate(tqdm(data_loader, desc="test...")):
            (pred, labels) = ctx.forward(batch_data)
            label = labels[0]
            loss = loss_fn(pred, label)
            if cuda:
                pred = pred.cpu()
                label = label.cpu()
            else:
                label = label.clone()  # cpu mode need copy the target data
            all_pred.append(pred.detach().numpy())
            all_labels.append(label.detach().numpy())
            accuracy = (torch.round(pred) == label).sum() / label.shape[0]
            accuracies.append(accuracy)
            losses.append(loss)

        all_pred, all_labels = np.concatenate(all_pred), np.concatenate(all_labels)

        fpr, tpr, _th = metrics.roc_curve(all_labels, all_pred)
        test_auc = metrics.auc(fpr, tpr)

        test_accuracies = torch.mean(torch.tensor(accuracies))
        test_loss = torch.mean(torch.tensor(losses))
        logger.info(
            f"test auc is {test_auc} accuracy is {test_accuracies}, loss is {test_loss}"
        )

    model.train()

    return test_auc


if __name__ == "__main__":

    reproducible = bool(int(os.environ.get("REPRODUCIBLE", 0)))
    embedding_staleness = int(os.environ.get("EMBEDDING_STALENESS", 10))

    if reproducible:
        setup_seed(3)

    model = DNN()
    logger.info("init Simple DNN model...")
    rank, device_id, world_size = get_rank(), get_local_rank(), get_world_size()

    mixed_precision = True
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(device_id)
        model.cuda(device_id)
    else:
        mixed_precision = False
        device_id = None

    logger.info(f"device_id is {device_id}")

    dense_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    embedding_optimizer = Adagrad(lr=1e-2)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    buffer_size = 10
    test_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data/test.npz"
    )
    test_dataset = TestDataset(test_dir, batch_size=128)
    test_interval = 254 // world_size - 1

    with TrainCtx(
        model=model,
        embedding_optimizer=embedding_optimizer,
        dense_optimizer=dense_optimizer,
        mixed_precision=mixed_precision,
        device_id=device_id,
    ) as ctx:
        train_dataloader = DataLoader(
            StreamingDataset(buffer_size),
            reproducible=reproducible,
            embedding_staleness=embedding_staleness,
        )
        test_loader = DataLoader(test_dataset)

        logger.info("start to training...")
        for (batch_idx, data) in enumerate(train_dataloader):
            (output, labels) = ctx.forward(data)
            label = labels[0]
            loss = loss_fn(output, label)
            scaled_loss = ctx.backward(loss)
            accuracy = (torch.round(output) == label).sum() / label.shape[0]

            logger.info(
                f"current batch idx: {batch_idx} loss: {float(loss)} scaled_loss: {float(scaled_loss)} accuracy: {float(accuracy)}"
            )
            if batch_idx % test_interval == 0 and batch_idx != 0:
                test_auc = test(model, test_loader, use_cuda)

                checkpoint_dir = os.environ.get("PERSIA_CKPT_DIR", None)
                if checkpoint_dir is not None and rank == 0:
                    logger.info(f"dump checkpoint to {checkpoint_dir}")
                    ctx.dump_checkpoint(checkpoint_dir, with_jit_model=True)

                if world_size == 1 and reproducible and embedding_staleness == 1:
                    np.testing.assert_equal(
                        np.array([test_auc]),
                        np.array([GPU_TEST_AUC if use_cuda else CPU_TEST_AUC]),
                    )

                break
