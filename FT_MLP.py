# ft_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as ty
from torch import Tensor
import math
import mmd

# Assume mmd_rbf_noaccelerate is defined elsewhere or implement it here
# For simplicity, we'll leave it as a placeholder

def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)

class Tokenizer(nn.Module):
    def __init__(self, d_numerical: int, d_token: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))  # +1 for [CLS] token
        self.bias = nn.Parameter(torch.Tensor(d_numerical, d_token)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num: Tensor) -> Tensor:
        x = torch.cat([torch.ones(len(x_num), 1, device=x_num.device), x_num], dim=1)  # Add [CLS]
        x = self.weight[None] * x[:, :, None]
        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_token % n_heads == 0
        self.W_q = nn.Linear(d_token, d_token)
        self.W_k = nn.Linear(d_token, d_token)
        self.W_v = nn.Linear(d_token, d_token)
        self.W_out = nn.Linear(d_token, d_token)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(m.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads, n_tokens, d_head)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        q, k, v = map(self._reshape, [q, k, v])
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(k.shape[-1]), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ v
        x = x.reshape(-1, self.n_heads, x.shape[1], x.shape[2]).transpose(1, 2).reshape(-1, x.shape[1], self.n_heads * x.shape[2])
        return self.W_out(x)

class ARFTransformer(nn.Module):
    def __init__(self, input_dim=52, d_token=128, n_layers=3, n_heads=8, d_ffn=256, num_classes=2):
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical=input_dim, d_token=d_token)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm0': nn.LayerNorm(d_token),
                'attention': MultiheadAttention(d_token, n_heads, dropout=0.6),
                'norm1': nn.LayerNorm(d_token),
                'ffn': nn.Sequential(
                    nn.Linear(d_token, d_ffn * 2),  # For ReGLU
                    ReGLU(),
                    nn.Dropout(0.6),
                    nn.Linear(d_ffn, d_token)
                )
            }) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_token, num_classes)

    def forward(self, source, target=None):
        # Tokenize source
        source_tokens = self.tokenizer(source)  # [batch_size, n_tokens, d_token]

        # Transformer layers
        x = source_tokens
        for layer in self.layers:
            x_residual = layer['norm0'](x)
            x_residual = layer['attention'](x_residual)
            x = x + x_residual
            x_residual = layer['norm1'](x)
            x_residual = layer['ffn'](x_residual)
            x = x + x_residual

        # Extract [CLS] token and predict
        source_features = x[:, 0]  # [batch_size, d_token]
        source_pred = self.head(source_features)  # [batch_size, num_classes]

        # Compute MMD loss if target is provided
        mmd_loss = 0
        if self.training and target is not None:
            target_tokens = self.tokenizer(target)
            x = target_tokens
            for layer in self.layers:
                x_residual = layer['norm0'](x)
                x_residual = layer['attention'](x_residual)
                x = x + x_residual
                x_residual = layer['norm1'](x)
                x_residual = layer['ffn'](x_residual)
                x = x + x_residual
            target_features = x[:, 0]  # [batch_size, d_token]
            mmd_loss = mmd.mmd_rbf_noaccelerate(source_features, target_features)

        return source_pred, mmd_loss