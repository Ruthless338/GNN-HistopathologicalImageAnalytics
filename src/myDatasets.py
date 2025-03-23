import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

class HistoDataset(Dataset):
    def __init__(self, data_dir, domain='source'):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        