import torch
import torch.nn as nn
import torch.nn.functional as F


class ATIN_op(nn.Module):
    def __init__(self, num_vars, seq_len, kernel_size=3, eta=0.5):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.eta = eta

        self.conv_layer = nn.Conv1d(
            in_channels=num_vars,
            out_channels=num_vars * 64,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_vars
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, D, T]  (D = num_vars)
        B, D, T = x.shape

        feat = self.conv_layer(x)  # Shape: [B, D * 64, T]

        feat = feat.reshape(B, D, 64, T).permute(0, 1, 3, 2)  # Shape: [B, D, T, 64]

        scores = self.attention(feat).squeeze(-1)  # Shape: [B, D, T]

        hard_mask = (scores > self.eta).float()

        mask = hard_mask - scores.detach() + scores

        # Shape: [B, D, T]
        return mask

