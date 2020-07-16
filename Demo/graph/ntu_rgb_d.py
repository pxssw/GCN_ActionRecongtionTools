import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools
# num_node = 25
num_node = 14
self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_ori_index = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                    (1, 11), (11, 12), (12, 13), (1, 8), (8, 9), (9, 10)]
# 'skeleton': [
#     [0, 1],  # head_top, upper_neck
#     [1, 2],  # upper_neck, r_shoulder
#     [2, 3],  # r_shoulder, r_elbow
#     [3, 4],  # r_elbow， r_wrist
#     [1, 5],  # upper_neck, l_shoulder
#     [5, 6],  # l_shoulder， l_elbow
#     [6, 7],  # l_elbow， l_wrist
#     [8, 9],  # r_hip， r_knee
#     [9, 10],  # r_knee， r_ankle
#     [11, 12],  # l_hip， l_knee
#     [12, 13],  # l_knee， l_ankle
# ],
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)

class Graph:
    def __init__(self, *args, **kwargs):
        self.inward = inward
        self.edge = self_link + inward

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
