import persia_embedding_py_cpu_client_sharded_server as pcc
import numpy as np

port = 8080
server = pcc.PyPersiaMessageQueueServer(port, 10)


def create_data(
    bs: int, dense_num: int = 2, target_num=2, sparse_num: int = 0, meta_num: int = 3
):
    denses, targets, sparses, metas = [], [], [], []
    for i in range(dense_num):
        denses.append(
            ("dense_{}".format(i), np.random.rand(bs, 128).astype(np.float32))
        )

    for i in range(target_num):
        targets.append(np.random.rand(bs, 1).astype(np.float32))

    # for i in range(sparse_num):
    # sparses.append([])

    for i in range(meta_num):
        metas.append(("meta_{}".format(i), np.random.rand(bs, 8).astype(np.float32)))
    return denses, targets, metas, sparses


middleware_addr = []
output_addr = [f"localhost:{port}"]
pcc.init(middleware_addr, output_addr)

batch = pcc.create_batch()
denses, targets, sparses, metas = create_data(128)

for (name, dense) in denses:
    print(dense.dtype, dense.shape)
    batch.add_dense_data(name, dense)

for target in targets:
    batch.add_target_data(target)

# for sparse in sparses:
# batch.add_sparse_data()
# ...

for (name, meta) in metas:
    # sparse: batch.add_meta_sparse_data()
    dense: batch.add_meta_dense_data(meta)

pcc.send_batch(batch)

bytes = server.recv()
from IPython import embed
embed()