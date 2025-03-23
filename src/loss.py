import torch.nn as nn
import torch



def mmd_loss(features_s, features_t, kernel_mul=2.0, kernel_num=5):
    '''
    计算多核MMD损失(基于高斯核)
    features_s:源域特征 [batch_size, feature_dim]
    features_t:目标域特征 [batch_size, feature_dim]
    kernel_mul:核函数的倍数
    kernel_num:核函数的数量
    '''
    batch_size = features_s.shape[0]
    features = torch.cat([features_s, features_t],dim=0)
    # 计算特征之间的欧氏距离
    features_0 = features.unsqueeze(0)
    features_1 = features.unsqueeze(1)
    l2_distance = ((features_0 - features_1)**2).sum(2)
    # 自适应计算基准带宽
    bandwidth = torch.sum(l2_distance) / (batch_size*2 * (batch_size*2-1))
    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    mmd = 0 
    for bw in bandwidth_list:
        kernel_val = torch.exp(-l2_distance / bw)
        kernel_xx = kernel_val[:batch_size, :batch_size]
        kernel_yy = kernel_val[batch_size:, batch_size:]
        kernel_xy = kernel_val[:batch_size, batch_size:]
        kernel_yx = kernel_val[batch_size:, :batch_size]

        mmd += torch.mean(kernel_xx) + torch.mean(kernel_yy) - torch.mean(kernel_xy) - torch.mean(kernel_yx)

    return mmd / kernel_num



def graph_loss(features, labels, d=2):
    '''
    同类样本拉进，异类样本拉远
    sqrt(d)是异类行本最短的距离
    '''
    loss = 0.0
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if labels[i] == labels[j] and labels[i] != -1:
                loss += torch.norm(features[i] - features[j]) ** 2
            elif labels[i] != labels[j] and labels[i] != -1 and labels[j] != -1:
                loss += torch.clamp(d - torch.norm(features[i] - features[j]) ** 2, min=0)
    return loss / (len(features) * (len(features) - 1) / 2)

'''
total_loss = mmd_loss + graph_loss + ce_loss
pseudo_labels是伪标签
'''
def loss(y_s_pred, y_t_pred, features_s, features_t, source_labels, pseudo_labels):
    # MMD损失
    mmd_loss = mmd_loss(features_s, features_t)
    # 图损失
    features = torch.cat([features_s, features_t], dim=0)
    labels = torch.cat([source_labels, pseudo_labels], dim=0)
    graph_loss = graph_loss(features, labels)
    # 交叉熵损失
    ce_loss = nn.CrossEntropyLoss()(y_s_pred, source_labels)
    if torch.sum(pseudo_labels != -1) > 0:
        ce_loss += nn.CrossEntropyLoss()(y_t_pred[pseudo_labels != -1], pseudo_labels[pseudo_labels != -1])

    loss = mmd_loss * 0.5 + graph_loss * 0.3 + ce_loss
    return loss
    