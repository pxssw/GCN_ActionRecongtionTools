import sys
sys.path.insert(0, '')
# print(sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.graph.tools import k_adjacency, normalize_adjacency_matrix
from model.mlp import MLP
from model.activation import activation_factory
import ipdb

class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            # A：18*18 -> (18*8=144)*18
            # ipdb.set_trace()
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = torch.Tensor(A_powers)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            # nn.init.uniform(tensor, a, b): fills the tensor with values drawn from the uniform distribution u(a,b)
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        # self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def forward(self, x):
        # ipdb.set_trace()
        N, C, T, V = x.shape
        self.A_powers = self.A_powers.to(x.device)
        A = self.A_powers.to(x.dtype)
        if self.use_mask:
            A = A + self.A_res.to(x.dtype)
        # A: 144*18 X:64*3*300*18  vu, AX = 64*3*300*144
        # support = torch.einsum('vu,nctu->nctv', A, x)
        # support = support.view(N, C, T, self.num_scales, V)
        # out = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        # onnx
        A = A.permute(1, 0).contiguous()
        x = x.permute(0, 2, 1, 3).contiguous()
        support = torch.matmul(x, A)
        support = support.view(N, T, C, self.num_scales, V)
        out = support.permute(0,3,2,1,4).contiguous().view(N, self.num_scales*C, T, V)
        # onnx
        # out = self.mlp(support)
        # ipdb.set_trace()
        return out


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16,3,30,25))
