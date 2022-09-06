from torch import nn
import torch

unpadded = [torch.Tensor((2,3)), torch.Tensor((2,3,3))]
padded = nn.utils.rnn.pad_sequence(unpadded, batch_first=True)