import os
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F



def load_aggregate_model(model, checkpoint_path, fine_tune):
    """
    If a model builds upon another one, then we load the other models weights here.

    :param model: to load the pre-trained weights for
    :param checkpoint_path: to the checkpoint of the pre-trained other model
    :param fine_tune: if to enable training for the pre-trained model weights
    """
    if not os.path.exists(checkpoint_path):
        raise Exception("Cannot find model checkpoint at %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False


def pad_inputs(inputs: list, max_length: int, pad_value, dtype, device):
    """
    Pad the inputs to a fixed size. The entries in the batch must be shorter than max_length.

    :param inputs: to pad to max_length (not just to the longest in the batch)
    :param max_length: the length upto which padding is applied
    :param pad_value: the value to use for padding
    :param dtype: of the padded inputs
    :param device: of the padded inputs

    :return: the padded inputs
    """
    max_size = inputs[0].size()
    trailing_dims = max_size[1:]
    out_dims = (len(inputs), max_length) + trailing_dims
    inputs_padded = torch.full(size=out_dims, fill_value=pad_value, dtype=dtype, device=device)
    # We need to iterate here, because each tensor has individual length
    for idx, tensor in enumerate(inputs):
        length = len(tensor)
        inputs_padded[idx, :length, ...] = tensor
    return inputs_padded
