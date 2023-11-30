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
            transforms.Resize((64, 192)),
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
        # print(image)
        # print(label)
        return image, label

    @staticmethod
    def format_label(label_str):
        character_dict = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
            'A': 10,
            'B': 11,
            'C': 12,
            'D': 13,
            'E': 14,
            'F': 15,
            'G': 16,
            'H': 17,
            'I': 18,
            'J': 19,
            'K': 20,
            'L': 21,
            'M': 22,
            'N': 23,
            'P': 24,
            'Q': 25,
            'R': 26,
            'S': 27,
            'T': 28,
            'U': 29,
            'V': 30,
            'W': 31,
            'X': 32,
            'Y': 33,
            'Z': 34,
            'NULL': 35
        }
        label_list = list(label_str) + ['NULL'] * (11 - len(label_str))
        label_list_classnum = []
        for character in label_list:
            label_list_classnum.append(character_dict[character])
        return label_list_classnum[:11]