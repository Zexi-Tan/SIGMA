import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F


def calculate_contrastive(fused_data, predicted_labels, tau=0.5):

    device = fused_data.device


    if not isinstance(predicted_labels, torch.Tensor):
        predicted_labels = torch.tensor(predicted_labels, dtype=torch.long, device=device)
    else:
        predicted_labels = predicted_labels.to(device).long()

    fused_data = torch.nan_to_num(fused_data, nan=0.0, posinf=1e4, neginf=-1e4)


    fused_data_norm = F.normalize(fused_data, p=2, dim=1, eps=1e-8)


    unique_labels = torch.unique(predicted_labels)


    if len(unique_labels) <= 1:
        return torch.tensor(0.0, requires_grad=True, device=device)


    prototypes = []
    for c in unique_labels:
        mask_c = (predicted_labels == c)

        proto_c = fused_data_norm[mask_c].mean(dim=0)

        proto_c = F.normalize(proto_c, p=2, dim=0)
        prototypes.append(proto_c)

    prototypes = torch.stack(prototypes)  # Shape: (C_batch, Hidden_Dims)


    # Shape: (N, C_batch)
    sim_matrix = torch.matmul(fused_data_norm, prototypes.T) / tau


    label_to_idx = {val.item(): idx for idx, val in enumerate(unique_labels)}
    target_indices = torch.tensor([label_to_idx[lbl.item()] for lbl in predicted_labels], device=device)


    loss = F.cross_entropy(sim_matrix, target_indices)

    return loss
