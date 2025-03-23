import torch

'''
生成伪标签,当标签概率大于epsilon时,标记为真实标签,否则标记为-1
'''
def generate_pseudo_labels(y_t_pred, epsilon=0.97):
    max_pred, pseudo_labels = torch.max(y_t_pred, dim=1)
    pseudo_labels[max_pred < epsilon] = -1  # -1表示未标记
    return pseudo_labels