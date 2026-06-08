import torch
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kmeans_gpu import kmeans

def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def sf_kmeans(matrix, device, num_clusters):
    matrix = torch.tensor(matrix, dtype=torch.float32)
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        matrix = torch.where(torch.isnan(matrix), torch.zeros_like(matrix), matrix)
        matrix = torch.where(torch.isinf(matrix), torch.zeros_like(matrix), matrix)

    tol=1e-4

    cluster_ids_x, _ = kmeans(
        X=matrix, num_clusters=num_clusters, distance='euclidean',  tol=tol , device=device
    )


    return cluster_ids_x



