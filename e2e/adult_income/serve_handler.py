from abc import ABC

import torch

from persia.ctx import InferCtx
from persia.service import get_embedding_worker_services
from ts.torch_handler.base_handler import BaseHandler

from ts.torch_handler.base_handler import BaseHandler

device_id = 0 if torch.cuda.is_available() else None


class PersiaHandler(BaseHandler, ABC):
    def initialize(self, context):
        super().initialize(context)
        embedding_worker_addrs = get_embedding_worker_services()
        self.persia_context = InferCtx(embedding_worker_addrs, device_id=device_id)
        self.persia_context.wait_for_serving()

    def preprocess(self, data):
        batch = data[0].get("batch")
        batch = bytes(batch)
        batch = self.persia_context.get_embedding_from_bytes(batch, device_id)

        model_input = self.persia_context.prepare_features(batch)
        return model_input

    def inference(self, data, *args, **kwargs):
        non_id_type_tensors, id_type_tensors, _ = data
        with torch.no_grad():
            results = self.model(non_id_type_tensors, id_type_tensors)
        return results

    def postprocess(self, data):
        data = torch.reshape(data, (-1,))
        data = data.tolist()
        return [data]
