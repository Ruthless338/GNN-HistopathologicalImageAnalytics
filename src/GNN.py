from ResNet import ResNet18
import torch
import torch.nn as nn
import dgl  # Deep Graph Library

# 利用ResNet18提取特征
class Backbone(nn.Module):
    def __init__(self, num_classes=1):
        super(Backbone, self).__init__()
        self.backbone = ResNet18()
        # 去掉最后一层全连接层
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.theta1 = nn.parameter(torch.rand(1))
        self.theta2 = nn.parameter(torch.rand(1))

    # g是图，features是节点特征
    # f = theta1 * W * x + theta2 * W * sum(x_neigh)
    def forward(self, g, features):
        # 图局部操作，避免对原始图的修改
        with g.local_scope():
            g.ndata['h'] = features
            # 计算图结点的邻居节点的聚合特征
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h = self.theta1 * torch.relu(self.linear(features)) + self.theta2 * torch.relu(self.linear(h_neigh))
            return h

# 领域自适应模型
class DomainAdaptationModel(nn.Module):
    def __init__(self, num_classes):
        super(DomainAdaptationModel, self).__init__()
        self.backbone = Backbone()
        self.gnn = GNN(512, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, features_s, features_t, graph=None):
        if graph is not None:
            features = torch.cat([features_s, features_t], dim=0)
            features_gnn = self.gnn(graph, features)
            features_s_gnn, features_t_gnn = torch.split(features_gnn, [features_s.shape[0], features_t.shape[0]], dim=0)
        else:
            features_s_gnn, features_t_gnn = features_s, features_t

        y_s, y_t = self.classifier(features_s_gnn), self.classifier(features_t_gnn)
        return y_s, y_t