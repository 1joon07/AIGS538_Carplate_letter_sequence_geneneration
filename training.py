import dataset as data
from torch.utils.data import DataLoader
import model as md
import torch.nn as nn
import cv2 as cv
import torch
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = data.LicensePlateDataset()
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = md.CNN(init_weights=True)
model.to(DEVICE)


for images, labels in train_loader:
    images = images.to(DEVICE)
    output = model(images)
    print(output)
    break