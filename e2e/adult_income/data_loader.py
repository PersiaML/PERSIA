import os

from tqdm import tqdm

from persia.embedding.data import PersiaBatch
from persia.logger import get_logger
from persia.ctx import DataCtx
from persia.utils import setup_seed

from data_generator import make_dataloader

logger = get_logger("data_loader")

setup_seed(3)

train_filepath = os.path.join("/data/", "train.npz")

if __name__ == "__main__":
    with DataCtx() as ctx:
        loader = make_dataloader(train_filepath)
        for (non_id_type_feature, id_type_features, label) in tqdm(
            loader, desc="generating data..."
        ):
            persia_batch = PersiaBatch(
                id_type_features,
                non_id_type_features=[non_id_type_feature],
                labels=[label],
            )
            ctx.send_data(persia_batch)
