import torch
from torch import nn
import torch.nn.functional as F
import math

from cov import TransformerEncoder1D, PositionalEncoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SIGMA(nn.Module):
    def __init__(self, input_dims, output_dims, seq_len, num_views=2, hidden_dims=64, dropout=0.2, top_k=2,
                 patch_size=8, target_keep_rate=0.35):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_views = num_views
        self.top_k = top_k
        self.target_keep_rate = target_keep_rate

        self.patch_size = patch_size

        self.input_stem = nn.Conv1d(
            in_channels=input_dims,
            out_channels=hidden_dims,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.pos_encoder = PositionalEncoding(d_model=hidden_dims)

        self.patch_scorer = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.GELU(),
            nn.Linear(hidden_dims // 2, 1)
        )

        self.view_encoders = nn.ModuleDict()
        self.view_projs = nn.ModuleDict()

        for i in range(self.num_views):
            self.view_encoders[f'view_{i}'] = TransformerEncoder1D(
                in_channels=hidden_dims,
                d_model=hidden_dims,
                num_layers=4,
                nhead=4,
                dropout=0.1
            )
            self.view_projs[f'view_{i}_proj'] = nn.Linear(hidden_dims, output_dims)

        self.repr_dropout = nn.Dropout(p=0.1)

        self.view_decoders = nn.ModuleDict()
        for i in range(self.num_views):
            self.view_decoders[f'view_{i}_decoder'] = nn.Sequential(
                nn.Linear(output_dims, hidden_dims * 2),
                nn.LayerNorm(hidden_dims * 2),
                nn.ReLU(),
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, input_dims * patch_size)
            )

        self.dropout = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dims))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, alpha=0.9, use_mask=True, force_keep_rate=None):
        B, T_len, D_feat = x.shape

        pad_len = (self.patch_size - T_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            T_len_padded = T_len + pad_len
        else:
            x_padded = x
            T_len_padded = T_len

        x_whole_list = []

        x_t = x_padded.transpose(1, 2)
        emb_base = self.input_stem(x_t).transpose(1, 2)
        emb_base = self.pos_encoder(emb_base)

        num_patches = emb_base.shape[1]

        patch_logits = self.patch_scorer(emb_base)
        patch_probs = torch.sigmoid(patch_logits)

        if use_mask:
            if force_keep_rate is not None:
                effective_keep_rate = force_keep_rate
            else:
                effective_keep_rate = self.target_keep_rate

            self.drop_rate = 1.0 - effective_keep_rate

            sparsity_loss = torch.tensor(0.0, device=x.device)
        else:
            self.drop_rate = 0.0
            effective_keep_rate = 1.0
            sparsity_loss = torch.tensor(0.0, device=x.device)

        for i in range(self.num_views):
            if use_mask:
                K = max(2, int(num_patches * effective_keep_rate))

                if self.training:
                    noise = torch.rand_like(patch_logits)
                    gumbel_noise = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
                    noisy_logits = patch_logits + gumbel_noise * (1.0 - alpha)
                else:
                    noisy_logits = patch_logits

                _, idx = torch.topk(noisy_logits, K, dim=1, largest=True, sorted=False)

                hard_mask = torch.zeros_like(patch_logits).scatter_(1, idx, 1.0)
                soft_mask = patch_probs

                if not self.training:
                    self.eval_mask = hard_mask.detach().clone()
                    self.eval_emb_base = emb_base.detach().clone()
                    self.eval_patch_probs = patch_probs.detach().clone()

                idx_expanded = idx.expand(-1, -1, self.hidden_dims)
                emb_gathered = torch.gather(emb_base, 1, idx_expanded)

                mask_gathered_soft = torch.gather(soft_mask, 1, idx)

                emb_gathered = emb_gathered * mask_gathered_soft

                emb_encoded = self.view_encoders[f'view_{i}'](emb_gathered.transpose(1, 2)).transpose(1, 2)

                global_context = emb_encoded.mean(dim=1, keepdim=True)  # [B, 1, hidden_dims]
                emb_full = global_context.expand(-1, num_patches, -1).clone()

                emb_full.scatter_(1, idx_expanded, emb_encoded)
                emb = emb_full

            else:
                emb = self.view_encoders[f'view_{i}'](emb_base.transpose(1, 2)).transpose(1, 2)

            emb = self.view_projs[f'view_{i}_proj'](emb)
            emb = self.repr_dropout(emb)
            x_whole_list.append(emb)

        diversity_loss = 0.0
        if self.training and self.num_views > 1:
            for i in range(self.num_views):
                for j in range(i + 1, self.num_views):
                    view_i, view_j = x_whole_list[i], x_whole_list[j]
                    B_v, N_v, D_v = view_i.shape
                    z_i, z_j = view_i.reshape(-1, D_v), view_j.reshape(-1, D_v)
                    var_i = z_i.var(dim=0, unbiased=False) + 1e-4
                    var_j = z_j.var(dim=0, unbiased=False) + 1e-4
                    z_i_norm = (z_i - z_i.mean(dim=0)) / torch.sqrt(var_i)
                    z_j_norm = (z_j - z_j.mean(dim=0)) / torch.sqrt(var_j)
                    c = torch.mm(z_i_norm.T, z_j_norm) / (B_v * N_v)
                    on_diag = torch.diagonal(c)
                    on_diag_loss = ((on_diag - 1) ** 2).sum()
                    off_diagonal_mask = ~torch.eye(D_v, dtype=torch.bool, device=c.device)
                    off_diag_loss = (c[off_diagonal_mask] ** 2).sum()
                    diversity_loss += on_diag_loss + 0.005 * off_diag_loss
            diversity_loss /= (self.num_views * (self.num_views - 1) / 2)

        reconstruction_loss = 0.0
        for i in range(self.num_views):
            view_recon_patches = self.view_decoders[f'view_{i}_decoder'](x_whole_list[i])
            view_recon_seq = view_recon_patches.reshape(B, num_patches * self.patch_size, self.input_dims)
            view_recon_seq = view_recon_seq[:, :T_len, :]
            view_recon_loss = F.smooth_l1_loss(view_recon_seq.float(), x.detach().float(), reduction='mean')
            reconstruction_loss += view_recon_loss

        if self.training:
            return x_whole_list, reconstruction_loss, diversity_loss, sparsity_loss
        else:
            return x_whole_list