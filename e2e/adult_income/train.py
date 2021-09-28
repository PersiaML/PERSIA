import os

from typing import Optional

import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics
import json

from persia.ctx import TrainCtx, eval_ctx, EmbeddingConfig
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


def test(
    model: torch.nn.Module,
    clear_embeddings: bool = False,
    checkpoint_dir: Optional[str] = None,
):
    logger.info("start to test...")
    model.eval()

    test_dir = os.path.join("/data/test.npz")
    test_dataset = TestDataset(test_dir, batch_size=128)

    with eval_ctx(model=model) as ctx:
        test_loader = Dataloder(test_dataset, is_training=False)
        if checkpoint_dir is not None:
            ctx.load_checkpoint(checkpoint_dir)
        accuracies, losses = [], []
        all_pred, all_target = [], []
        for (batch_idx, batch_data) in enumerate(tqdm(test_loader, desc="test...")):
            (output, target) = ctx.forward(batch_data)
            loss = loss_fn(output, target)
            all_pred.append(output.cpu().detach().numpy())
            all_target.append(target.cpu().detach().numpy())
            accuracy = (torch.round(output) == target).sum() / target.shape[0]
            accuracies.append(accuracy)
            losses.append(float(loss))

        if clear_embeddings:
            ctx.clear_embeddings()
            num_ids = sum(ctx.get_embedding_size())
            assert num_ids == 0, f"clear embedding failed"

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

    eval_checkpoint_dir = os.environ["EVAL_CHECKPOINT_DIR"]
    infer_checkpoint_dir = os.environ["INFER_CHECKPOINT_DIR"]
    hdfs_checkpoint_dir = os.environ["HDFS_CHECKPOINT_DIR"]
    test_interval = 254
    buffer_size = 10

    embedding_config = EmbeddingConfig()
    with TrainCtx(
        model=model,
        sparse_optimizer=sparse_optimizer,
        dense_optimizer=dense_optimizer,
        device_id=device_id,
        embedding_config=embedding_config,
    ) as ctx:
        train_dataloader = Dataloder(
            StreamingDataset(buffer_size), reproducible=True, embedding_staleness=1
        )
        for (batch_idx, data) in enumerate(train_dataloader):
            (output, target) = ctx.forward(data)
            loss = loss_fn(output, target)
            scaled_loss = ctx.backward(loss)
            accuracy = (torch.round(output) == target).sum() / target.shape[0]
            logger.info(
                f"current idx: {batch_idx} loss: {loss} scaled_loss: {scaled_loss} accuracy: {accuracy}"
            )

            if batch_idx % test_interval == 0 and batch_idx != 0:
                test_auc, test_acc = test(model)
                np.testing.assert_equal(
                    np.array([test_auc]), np.array([0.8934601372796367])
                )
                break

        ctx.dump_checkpoint(eval_checkpoint_dir)
        logger.info(f"dump checkpoint to {eval_checkpoint_dir}")

        ctx.dump_checkpoint(hdfs_checkpoint_dir)
        logger.info(f"dump checkpoint to {hdfs_checkpoint_dir}")

        ctx.dump_checkpoint(infer_checkpoint_dir, with_jit_model=True)
        logger.info(f"dump checkpoint to {infer_checkpoint_dir}")

        ctx.clear_embeddings()
        num_ids = sum(ctx.get_embedding_size())
        assert num_ids == 0, f"clear embedding failed"

    eval_auc, eval_acc = test(
        model, clear_embeddings=True, checkpoint_dir=eval_checkpoint_dir
    )
    np.testing.assert_equal(np.array([test_auc]), np.array([eval_auc]))

    eval_auc, eval_acc = test(
        model, clear_embeddings=True, checkpoint_dir=hdfs_checkpoint_dir
    )
    np.testing.assert_equal(np.array([test_auc]), np.array([eval_auc]))

    result_filepath = os.environ["RESULT_FILE_PATH"]
    result = {
        "test_auc": test_auc,
        "eval_auc": eval_auc,
    }
    result = json.dumps(result)

    with open(result_filepath, "w") as f:
        f.write(result)
