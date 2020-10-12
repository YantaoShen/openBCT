from __future__ import absolute_import
import torch


def extract_cnn_feature(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        outputs = outputs.cpu()
        return outputs
