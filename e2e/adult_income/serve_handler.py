from persia.ctx import InferCtx
from persia.service import get_middleware_services
from ts.torch_handler.base_handler import BaseHandler

from abc import ABC
import torch
import os


class PersiaHandler(BaseHandler, ABC):
    def initialize(self, context):
        super().initialize(context)
        middleware_addrs = get_middleware_services()
        self.persia_context = InferCtx(middleware_addrs)

    def preprocess(self, data):
        batch = data[0].get("batch")
        batch = bytes(batch)
        batch = self.persia_context.get_embedding_from_bytes(batch, 0)

        model_input = self.persia_context.prepare_features(batch)
        return model_input

    def inference(self, data, *args, **kwargs):
        denses, sparses, _ = data
        with torch.no_grad():
            results = self.model(denses, sparses)
        return results

    def postprocess(self, data):
        data = torch.reshape(data, (-1,))
        data = data.tolist()
        return [data]
