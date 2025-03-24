from buildGraph import build_graph
from loss import loss
from GNN import DomainAdaptationModel
from pseudoLabels import generate_pseudo_labels
import torch
from myDatasets import myDataset
from torch.utils.data import DataLoader
from visualize import visualize_features
from matplotlib import pyplot as plt

num_epochs = 100
threshold = 150 #图构建的阈值
epsilon = 0.97 #伪标签阈值
lr = 0.001
num_classes = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DomainAdaptationModel(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

source_dataset = myDataset(data_dir='', domain='train', is_train=True, is_transform=True)
target_dataset = myDataset(data_dir='', domain='train', is_train=True,is_transform=True)
target_val_dataset = myDataset(data_dir='', domain='val', is_train=False,is_transform=False)

source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
target_val_loader = DataLoader(target_val_dataset, batch_size=16, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    loss_values = []
    for (src_imgs, src_labels),(tgt_imgs, _) in zip(source_loader, target_loader):
        src_imgs = src_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)
        src_labels = src_labels.to(device)
        # 提取特征
        features_s = model.backbone(src_imgs)
        features_t = model.backbone(tgt_imgs)
        # 建图
        graph = build_graph(features_s.detach(), features_t.detach(), threshold=threshold) 
        graph = graph.to(device)
        # 前向传播
        y_s, y_t = model(features_s, features_t, graph)
        # 生成伪标签
        with torch.no_grad():
            pseudo_labels = generate_pseudo_labels(y_t,epsilon=epsilon)
        # 损失函数计算
        loss = loss(y_s, y_t, features_s, features_t, src_labels, pseudo_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

    # 损失函数曲线绘制
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 每个epoch做验证,伪标签置信度
    model.eval()
    total_conf = 0.0
    num_samples = target_val_dataset.__len__()
    with torch.no_gard():
        for tgt_imgs, _ in target_val_loader:
            tgt_imgs = tgt_imgs.to(device)
            features_t = model.backbone(tgt_imgs)
            graph = build_graph(features_t.detach(), features_t.detach(), threshold=threshold)
            _, y_t = model(features_t, features_t, graph)
            probs = torch.softmax(y_t,dim=1)
            max_probs, _ = torch.max(probs,dim=1)
            total_conf += torch.sum(max_probs).item()
        avg_conf = total_conf / num_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Confidence: {avg_conf:.4f}')
    
    torch.save(model.state_dict(),'model.pth')
    torch.save(optimizer.state_dict(),'optimizer.pth')

# 可视化特征分布(源域训练集+目标域验证集)
model.eval()
src_features, tgt_features = [], []
with torch.no_grad():
    for imgs, _ in source_loader:
        features = model.backbone(imgs.to(device))
        src_features.append(features)
    for imgs, _ in target_val_loader:
        features = model.backbone(imgs.to(device))
        tgt_features.append(features)
src_features = torch.cat(src_features, dim=0)
tgt_features = torch.cat(tgt_features, dim=0)

# 绘制特征分布
plt.figure(figsize=(10,6))
visualize_features(src_features, 'Source')
visualize_features(tgt_features, 'Target')
plt.title('Unsupervised Feature Distribution')
plt.legend()
plt.show()