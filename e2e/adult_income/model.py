from typing import List

import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(
        self, dense_mlp_output_size: int = 16, sparse_mlp_output_size: int = 128
    ):
        super(DNN, self).__init__()

        self.dense_mlp = torch.nn.Linear(5, dense_mlp_output_size)
        self.dense_bn = nn.BatchNorm1d(dense_mlp_output_size)

        self.sparse_mlp = torch.nn.Linear(64, sparse_mlp_output_size)
        self.sparse_bn = nn.BatchNorm1d(sparse_mlp_output_size)

        self.ln1 = nn.Linear(dense_mlp_output_size + sparse_mlp_output_size, 256)
        self.ln2 = nn.Linear(256, 128)
        self.ln3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dense_x: torch.Tensor, sparses: List[torch.Tensor]):
        sparse_concat = torch.cat(sparses, dim=1)
        sparse = self.sparse_mlp(sparse_concat.float())
        sparse = self.sparse_bn(sparse)

        dense_x = self.dense_mlp(dense_x)
        dense_x = self.dense_bn(dense_x)
        x = torch.cat([sparse, dense_x], dim=1)
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)

        return self.sigmoid(x)


if __name__ == "__main__":
    model = DNN()
    batch_size = 64
    dense = torch.ones(batch_size, 5)
    sparses = [torch.ones(batch_size, 8) for _ in range(8)]
    output = model(dense, sparses)
    print(output)
