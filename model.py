import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class STLayer(nn.Module):
#     def __init__(self, init_weights=True):
#         super(STLayer, self).__init__()
#
#         self.conv1 = nn.Sequential(nn.Conv2d(1, 48, kernel_size=(5, 5), stride=1, padding=0), nn.BatchNorm2d(48), nn.ReLU())
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#
#         self.conv2 = nn.Sequential(nn.Conv2d(48, 32, kernel_size=(5, 5), stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU())
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#
#         self.fc1 = nn.Sequential(nn.Linear(in_features=32 * 5 * 21, out_features=50), nn.BatchNorm1d(50), nn.ReLU())
#         self.fc2 = nn.Sequential(nn.Linear(in_features=50, out_features=6), nn.BatchNorm1d(6), nn.ReLU())
#
#
#         if init_weights:
#             self.init_weights_function()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x, p1 = self.pool1(x)
#
#         x = self.conv2(x)
#         x, p2 = self.pool2(x)
#         print(x.size())
#         x = x.view(x.size(0), -1)
#         print(x.size())
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return x
#
#     def init_weights_function(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)




class CNN(nn.Module):
    def __init__(self, init_weights=True):
        super(CNN, self).__init__()

        self.localization = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=48, kernel_size=5, stride=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, stride=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                nn.Flatten(),
                nn.Linear(in_features=32 * 5 * 21, out_features=50),
                nn.ReLU(),
                nn.Linear(in_features=50, out_features=6),
                nn.ReLU()

        )

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(5, 5), stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0), nn.BatchNorm2d(256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(in_features=5120, out_features=1024), nn.ReLU())
        
        self.parallel_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features=1024, out_features=37), nn.Softmax())
            for _ in range(11)
        ])

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        if init_weights:
            self.init_weights_function()



    def stlayer(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stlayer(x)
        x = self.conv1(x)
        x, p1 = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x, p2 = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x, p3 = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        l = []
        for layer in self.parallel_layers:
            l.append(layer(x))

        return l

    def init_weights_function(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
