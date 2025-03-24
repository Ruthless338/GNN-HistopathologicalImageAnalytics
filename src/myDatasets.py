import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

class myDataset(Dataset):
    def __init__(self, data_dir, domain='source', is_train=True, is_transform=True):
        self.data_dir = os.path.join(data_dir, domain)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.is_transform = is_transform
        self.samples = []
        self.labels = []
        self.data_dir = os.path.join(data_dir, domain)
        if is_train:
            self.data_dir = os.path.join(self.data_dir, 'train')
        else:
            self.data_dir = os.path.join(self.data_dir, 'val')
        # 源域胶质瘤病变标签值为1,正常标签值为0
        if domain == 'source':
            for class_name in os.listdir(self.data_dir):
                folder_path = os.path.join(self.data_dir, class_name)
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    self.samples.append(file_path)
                    self.labels.append(1 if class_name == 'glioma' else 0)
        # 目标域标签值恒为-1
        else:
            for folder_name in os.listdir(self.data_dir):
                folder_path = os.path.join(self.data_dir, folder_name)
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    self.samples.append(file_path)
                    self.labels.append(-1)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.is_transform:
            img = self.transform(img)
        return img, label