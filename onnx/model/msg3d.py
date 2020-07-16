import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP, MLP2
from model.activation import activation_factory
# from tools.utils.graph import STGCN_Graph
from ptflops import get_model_complexity_info

import ipdb


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        # self.stgcn_graph = STGCN_Graph(layout='openpose',strategy='spatial')
        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 64
        c2 = c1 * 1     # 64
        # c3 = c2 * 2     # 384

        mlp1 = MLP(3 * num_gcn_scales, [c1], dropout=0, activation='relu')
        mlp2 = MLP2(c1 * num_gcn_scales, [c2], dropout=0, activation='relu')
        # r=3 STGC blocks
        # self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            mlp1,
            MS_TCN(c1, c1, kernel_size=5),
            MS_TCN(c1, c1, kernel_size=5))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1, kernel_size=5)

        # self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            mlp2,
            MS_TCN(c2, c2, kernel_size=5, stride=2),
            MS_TCN(c2, c2, kernel_size=5))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2, kernel_size=5)

        # self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        # self.sgcn3 = nn.Sequential(
        #     MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
        #     MS_TCN(c2, c3, stride=2),
        #     MS_TCN(c3, c3))
        # self.sgcn3[-1].act = nn.Identity()
        # self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c2, num_class)
        # self.fc = nn.Linear(c3, num_class)
        # self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # ipdb.set_trace()
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        # x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = F.relu(self.sgcn1(x), inplace=True)
        x = self.tcn1(x)

        # x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = F.relu(self.sgcn2(x), inplace=True)
        x = self.tcn2(x)

        # ipdb.set_trace()
        # x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        # x = F.relu(self.sgcn3(x), inplace=True)
        # x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out

import time
if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=11,
        num_point=14,
        num_person=1,
        num_gcn_scales=8,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    ).cuda()
    flops, params = get_model_complexity_info(model, (3, 10, 14, 1), as_strings=True, print_per_layer_stat=True)
    print("%s |%s |%s" % ('MSG3D', flops, params))
    N, C, T, V, M = 1, 3, 10, 14, 1
    seed = 1
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sum = 0
    for i in range(10):
        x = torch.rand(N,C,T,V,M).cuda()
        # torch.cuda.synchronize()
        tmp1 = time.time()
        model.forward(x)
        # torch.cuda.synchronize()
        tmp2 = time.time()
        sum += tmp2-tmp1
    print("inference time: {}".format(sum/10.0))