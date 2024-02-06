import numpy as np
import torch
import torch.nn as nn


class C4Net(nn.Module):

    def __init__(self):
        super(C4Net, self).__init__()
        # Define Model Layers Here
        pass

    def forward(self, x):
        # x is the input, should be 3x7x6, 3 channels by 7 rows by 6 columns

        # Use the model layers
        # return a 7 len tensor and a 1x1 tensor
        return torch.tensor([0, 0, 0, 0, 0, 0, 0]), torch.tensor([0])


def test():
    ch, R, C = 3, 7, 6
    rand = torch.randn((ch, R, C))
    net = C4Net()
    p, v = net(rand)
    assert p.shape == torch.Size([7])
    assert v.shape == torch.Size([1])


test()
