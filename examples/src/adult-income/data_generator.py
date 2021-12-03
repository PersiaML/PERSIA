from typing import Iterable, List, Tuple

import numpy as np

from tqdm import tqdm

from persia.embedding.data import IDTypeFeature, NonIDTypeFeature, Label


class DataLoader:
    def __init__(
        self,
        non_id_type_feature_data,
        id_type_feature_data,
        label_data,
        id_type_feature_names,
        batch_size: int = 128,
        skip_last_batch: bool = False,
    ):
        self.non_id_type_feature_data = non_id_type_feature_data
        self.id_type_feature_data = id_type_feature_data
        self.label_data = label_data
        self.id_type_feature_names = id_type_feature_names
        self.batch_size = batch_size
        self.skip_last_batch = skip_last_batch

        dataset_size = len(label_data)
        loader_size = (dataset_size - 1) // batch_size + 1
        if skip_last_batch:
            loader_size = loader_size - 1

        self.loader_size = loader_size

    def __iter__(
        self,
    ) -> Iterable[Tuple[NonIDTypeFeature, List[IDTypeFeature], Label]]:
        batch_size = self.batch_size
        skip_last_batch = self.skip_last_batch
        id_type_feature_names = self.id_type_feature_names
        id_type_feature_data = self.id_type_feature_data
        label_data = self.label_data

        dataset_size = len(self.label_data)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            if end == dataset_size and skip_last_batch:
                print("skip last batch...")
                continue

            non_id_type_feature = NonIDTypeFeature(
                self.non_id_type_feature_data[start:end, :]
            )
            id_type_features = []

            for id_type_feature_idx, feature_name in enumerate(id_type_feature_names):
                id_type_feature = []
                id_type_feature_batch = id_type_feature_data[start:end]
                for batch_idx in range(len(id_type_feature_batch)):
                    id_type_feature.append(
                        id_type_feature_batch[batch_idx][
                            id_type_feature_idx : id_type_feature_idx + 1
                        ]
                    )
                id_type_features.append(IDTypeFeature(feature_name, id_type_feature))

            label = label_data[start:end]
            label = Label(label.reshape(len(label), -1))

            yield non_id_type_feature, id_type_features, label

    def __len__(self) -> int:
        return self.loader_size


def make_dataloader(
    data_filepath: str, batch_size: int = 128, skip_last_batch: bool = False
) -> DataLoader:
    with np.load(data_filepath) as data:
        target = data["target"]
        continuous_data = data["continuous_data"]
        categorical_data = data["categorical_data"]
        categorical_columns = data["categorical_columns"]

    return DataLoader(
        continuous_data,
        categorical_data,
        target,
        categorical_columns,
        batch_size,
        skip_last_batch,
    )


if __name__ == "__main__":
    loader = make_dataloader("data/train.npz", 128, skip_last_batch=False)
    for (dense, sparse, target) in tqdm(loader, desc="generate_data"):
        ...
