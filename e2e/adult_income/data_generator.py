import numpy as np

from tqdm import tqdm


def generate_loader(
    continuous_data,
    categorical_data,
    target,
    categorical_columns,
    batch_size: int = 128,
    skip_last_batch: bool = False,
):
    dataset_size = len(target)
    for start in range(0, dataset_size, batch_size):
        end = min(start + batch_size, dataset_size)
        if end == dataset_size and skip_last_batch:
            print("skip last batch...")
            continue
        continuous_batch_data = continuous_data[start:end, :]
        sparse_batch_data = []

        for idx, categorical_column in enumerate(categorical_columns):
            tmp_sparse_batch = []
            categorical_batch_data = categorical_data[start:end]
            for i in range(len(categorical_batch_data)):
                tmp_sparse_batch.append(categorical_batch_data[i][idx : idx + 1])
            sparse_batch_data.append((categorical_column, tmp_sparse_batch))

        target_batch = target[start:end]
        target_batch = target_batch.reshape(len(target_batch), -1)

        yield continuous_batch_data, sparse_batch_data, target_batch


def make_dataloader(
    data_filepath: str, batch_size: int = 128, skip_last_batch: bool = False
):
    with np.load(data_filepath) as data:
        target = data["target"]
        continuous_data = data["continuous_data"]
        categorical_data = data["categorical_data"]
        categorical_columns = data["categorical_columns"]

    dataset_size = len(target)
    loader_size = (dataset_size - 1) // batch_size + 1
    if skip_last_batch:
        loader_size = loader_size - 1

    return loader_size, generate_loader(
        continuous_data,
        categorical_data,
        target,
        categorical_columns,
        batch_size,
        skip_last_batch,
    )


if __name__ == "__main__":
    loader = make_dataloader("data_source/train.npz", 128, skip_last_batch=False)
    for (dense, sparse, target) in tqdm(loader, desc="generate_data"):
        ...
