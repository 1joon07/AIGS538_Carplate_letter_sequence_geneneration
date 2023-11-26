from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class LicensePlateDataset(Dataset):
    def __init__(self, directory='./CNN_generated_dataset', label_file='./CNN_generated_dataset/Labels.csv'):
        self.directory = directory
        self.labels = pd.read_csv(label_file, index_col='Filename').to_dict()['Label']
        self.images = [img for img in os.listdir(directory) if img.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 96)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path)
        image = self.transform(image)

        label_str = self.labels.get(img_name, "")
        label = self.format_label(label_str)
        # print(img_name)
        print(image)
        # print(label)
        return image, label

    @staticmethod
    def format_label(label_str):
        label_list = list(label_str) + ['NULL'] * (11 - len(label_str))
        return label_list[:11]