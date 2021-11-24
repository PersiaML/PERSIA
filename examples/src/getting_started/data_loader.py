import os

import numpy as np

from tqdm import tqdm

from persia.prelude import PersiaBatch
from persia.logger import get_logger
from persia.ctx import DataCtx
from persia.utils import setup_seed

from data_generator import make_dataloader

logger = get_logger("data_loader")

setup_seed(3)

train_filepath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data/train.npz"
)

logger.info("init py client done...")
if __name__ == "__main__":
    with DataCtx() as ctx:
        _, loader = make_dataloader(train_filepath)
        for (dense, batch_sparse_ids, target) in tqdm(loader, desc="gen batch data..."):
            batch_data = PersiaBatch()
            batch_data.add_non_id_type_feature([dense])
            batch_data.add_id_type_features(batch_sparse_ids)
            batch_data.add_label(target)
            ctx.send_data(batch_data)
