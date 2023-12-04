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
    def __init__(self, init_weights=False):
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

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),

                nn.Flatten(),
                nn.Linear(in_features=9216, out_features=100),
                nn.ReLU(),
                nn.Linear(in_features=100, out_features=6),
                nn.ReLU()

        )

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout4 = nn.Dropout(p=0.25)

        # self.conv5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout5 = nn.Dropout(p=0.25)

        self.fc1 = nn.Sequential(nn.Flatten())
        
        # self.parallel_layers = nn.ModuleList([
        #     nn.Sequential(nn.Linear(in_features=1024, out_features=37), nn.Softmax())
        #     for _ in range(11)
        # ])

        self.output0 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=36), nn.Softmax(dim=1))

        ## 100x100
        # self.output0 = nn.Sequential(nn.Linear(in_features=4608, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=36), nn.Softmax(dim=1))

        # self.output1 = nn.Sequential(nn.Linear(in_features=8448, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=36), nn.Softmax(dim=1))
        # self.output2 = nn.Sequential(nn.Linear(in_features=8448, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=36), nn.Softmax(dim=1))
        # self.output3 = nn.Sequential(nn.Linear(in_features=8448, out_features=1024), nn.ReLU(), nn.Linear(in_features=1024, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=36), nn.Softmax(dim=1))
        # self.output4 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))
        # self.output5 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))
        # self.output6 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))
        # self.output7 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))
        # self.output8 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))
        # self.output9 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))
        # self.output10 = nn.Sequential(nn.Linear(in_features=1024, out_features=36), nn.Softmax(dim=1))

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

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # x = self.stlayer(x)
        x = self.conv1(x)
        x = self.pool1(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        # x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        # x = self.dropout3(x)
        #
        x = self.conv4(x)
        x = self.pool4(x)
        # x = self.dropout4(x)
        #
        # x = self.conv5(x)
        # # x = self.pool5(x)
        # x = self.dropout5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # l = []
        # for layer in self.parallel_layers:
        #     l.append(layer(x))

        out0 = self.output0(x)
        # out1 = self.output1(x)
        # out2 = self.output2(x)
        # out3 = self.output3(x)
        # out4 = self.output4(x)
        # out5 = self.output5(x)
        # out6 = self.output6(x)
        # out7 = self.output7(x)
        # out8 = self.output8(x)
        # out9 = self.output9(x)
        # out10 = self.output10(x)


        # out = torch.cat([out0.unsqueeze(1), out1.unsqueeze(1), out2.unsqueeze(1),
        #              out3.unsqueeze(1), out4.unsqueeze(1), out5.unsqueeze(1),
        #              out6.unsqueeze(1), out7.unsqueeze(1), out8.unsqueeze(1),
        #              out9.unsqueeze(1), out10.unsqueeze(1)], dim=1)

        # out = torch.cat([out0.unsqueeze(1)], dim=1)
        # print(f'length {out.size()}')
        return out0


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
