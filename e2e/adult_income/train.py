import os
import json

from typing import Optional

import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics

from persia.ctx import TrainCtx, eval_ctx
from persia.distributed import DDPOption
from persia.embedding.optim import Adagrad
from persia.embedding.data import PersiaBatch
from persia.env import get_rank, get_local_rank, get_world_size
from persia.logger import get_default_logger
from persia.utils import setup_seed
from persia.data import Dataloder, PersiaDataset, StreamingDataset
from persia.prelude import PersiaBatchDataSender

from model import DNN
from data_generator import make_dataloader


logger = get_default_logger("nn_worker")

setup_seed(3)

CPU_TEST_AUC = 0.8936692224423999
GPU_TEST_AUC = 0.8934601372796367


class TestDataset(PersiaDataset):
    def __init__(self, test_dir: str, batch_size: int = 128):
        super(TestDataset, self).__init__(buffer_size=10)
        self.loader = make_dataloader(test_dir, batch_size)

        logger.info(f"test dataset size is {len(self.loader)}")

    def fetch_data(self, persia_sender_channel: PersiaBatchDataSender):
        logger.info("test loader start to generating data...")
        for _idx, (non_id_type_feature, id_type_features, label) in enumerate(
            tqdm(self.loader, desc="generating data")
        ):
            persia_batch = PersiaBatch(
                id_type_features,
                non_id_type_features=[non_id_type_feature],
                labels=[label],
                requires_grad=False,
            )
            persia_sender_channel.send(persia_batch.data)

    def __len__(self):
        return len(self.loader)


def test(
    model: torch.nn.Module,
    clear_embeddings: bool = False,
    checkpoint_dir: Optional[str] = None,
    cuda: bool = True,
):
    logger.info("start to test...")
    model.eval()

    test_dir = os.path.join("/data/test.npz")
    test_dataset = TestDataset(test_dir, batch_size=128)

    with eval_ctx(model=model) as ctx:
        test_loader = Dataloder(test_dataset, is_training=False)
        if checkpoint_dir is not None:
            logger.info(f"loading checkpoint {checkpoint_dir}")
            ctx.load_checkpoint(checkpoint_dir)

        accuracies, losses = [], []
        all_pred, all_target = [], []
        for (_batch_idx, batch_data) in enumerate(tqdm(test_loader, desc="test...")):
            (pred, targets) = ctx.forward(batch_data)
            target = targets[0]
            loss = loss_fn(pred, target)
            if cuda:
                pred = pred.cpu()
                target = target.cpu()
            else:
                # cpu mode need copy the target data to avoid use the expired data.
                target = target.clone()
            all_pred.append(pred.detach().numpy())
            all_target.append(target.detach().numpy())
            accuracy = (torch.round(pred) == target).sum() / target.shape[0]
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
    mixed_precision = True

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        model.cuda(device_id)
        backend = "nccl"
        cuda = True
    else:
        mixed_precision = False
        device_id = None
        backend = "gloo"
        cuda = False

    dense_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    embedding_optimizer = Adagrad(lr=1e-2)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    eval_checkpoint_dir = os.environ["EVAL_CHECKPOINT_DIR"]
    infer_checkpoint_dir = os.environ["INFER_CHECKPOINT_DIR"]
    hdfs_checkpoint_dir = os.environ["HDFS_CHECKPOINT_DIR"]
    test_interval = 254
    buffer_size = 10

    with TrainCtx(
        model=model,
        embedding_optimizer=embedding_optimizer,
        dense_optimizer=dense_optimizer,
        device_id=device_id,
        mixed_precision=mixed_precision,
        distributed_option=DDPOption(backend=backend),
    ) as ctx:
        train_dataloader = Dataloder(
            StreamingDataset(buffer_size), reproducible=True, embedding_staleness=1
        )
        for (batch_idx, data) in enumerate(train_dataloader):
            (output, targets) = ctx.forward(data)
            target = targets[0]

            loss = loss_fn(output, target)
            scaled_loss = ctx.backward(loss)
            accuracy = (torch.round(output) == target).sum() / target.shape[0]
            logger.info(
                f"current idx: {batch_idx} loss: {loss} scaled_loss: {scaled_loss} accuracy: {accuracy}"
            )

            if batch_idx % test_interval == 0 and batch_idx != 0:
                test_auc, test_acc = test(model, cuda=cuda)
                np.testing.assert_equal(
                    np.array([test_auc]),
                    np.array([GPU_TEST_AUC if cuda else CPU_TEST_AUC]),
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
        model, clear_embeddings=True, checkpoint_dir=eval_checkpoint_dir, cuda=cuda
    )
    np.testing.assert_equal(np.array([test_auc]), np.array([eval_auc]))

    eval_auc, eval_acc = test(
        model, clear_embeddings=True, checkpoint_dir=hdfs_checkpoint_dir, cuda=cuda
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
