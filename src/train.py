from buildGraph import build_graph
from loss import loss
from GNN import DomainAdaptationModel
from pseudoLabels import generate_pseudo_labels
import torch


num_epochs = 100
threshold = 150 #图构建的阈值
epsilon = 0.97 #伪标签阈值
lr = 0.001
num_classes = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DomainAdaptationModel(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()