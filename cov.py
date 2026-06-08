import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)\

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder1D(nn.Module):
    def __init__(self, in_channels, d_model, num_layers=4, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()


        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        # x shape: [Batch, d_model, Seq_Len_K]
        x = x.transpose(1, 2)  # [Batch, Seq_Len_K, d_model]
        x = self.input_proj(x)

        out = self.transformer(x)

        out = out.transpose(1, 2)  # [Batch, d_model, Seq_Len_K]
        return out