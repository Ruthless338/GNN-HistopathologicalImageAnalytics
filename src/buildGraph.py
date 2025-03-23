import torch
import numpy as np
import dgl
from sklearn.metrics.pairwise import euclidean_distances

'''
每个样本都被提取特征变成特征空间中的一个点，我们可以通过计算这些点之间的相似度来构建一个图，这个图的节点是样本
当两个样本之间的相似度小于某个阈值时，我们就在这两个样本之间连一条边，这样就构建了一个图
'''
def build_graph(features_s, features_t, threshold=150):
    features = torch.cat([features_s, features_t], dim=0).detach().cpu().numpy()
    dist_matrix = euclidean_distances(features) # 计算特征之间的欧氏距离
    edges = np.argwhere(dist_matrix < threshold)
    src, dst = edges[:, 0], edges[:, 1]
    g = dgl.graph((src, dst))
    return g