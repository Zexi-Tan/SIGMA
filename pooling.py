import numpy as np
import torch
import torch.nn.functional as F


def pooling(x_whole_list):
    pooled_list = []
    for x_view in x_whole_list:
        pooled = x_view.mean(dim=1)
        pooled_list.append(pooled)
    return pooled_list

