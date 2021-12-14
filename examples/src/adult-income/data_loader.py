import os

from tqdm import tqdm

from persia.embedding.data import PersiaBatch
from persia.logger import get_logger
from persia.ctx import DataCtx

from data_generator import make_dataloader

logger = get_logger("data_loader")

train_filepath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data/train.npz"
)

if __name__ == "__main__":
    with DataCtx() as ctx:
        loader = make_dataloader(train_filepath)
        for (non_id_type_feature, id_type_features, label) in tqdm(
            loader, desc="gen batch data..."
        ):
            persia_batch = PersiaBatch(
                id_type_features,
                non_id_type_features=[non_id_type_feature],
                labels=[label],
                requires_grad=True,
            )
            ctx.send_data(persia_batch)
