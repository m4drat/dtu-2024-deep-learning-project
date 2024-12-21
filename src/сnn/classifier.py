import numpy as np
import torch.nn as nn
import torch.nn.functional as F


kernel_size = 5

class CNNClassifier(nn.Module):
    def __init__(self, im_size, num_of_classes):
        super().__init__()

        # 1st convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size, padding=1)
        self.bnorm1 = nn.BatchNorm2d(64)
        conv1_size = int(self._get_cv_output_size(im_size, padding = 1)/2)
        print(f'Con1 size: {conv1_size * 2}')
        print(f'MaxPool1 size: {conv1_size}')


        # 2nd convolution layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size)
        self.bnorm2 = nn.BatchNorm2d(128)
        conv2_size = int(self._get_cv_output_size(conv1_size)/2)
        print(f'Con2 size: {conv2_size * 2}')
        print(f'MaxPool2 size: {conv2_size}')


        # 3d convolution layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size)
        self.bnorm3 = nn.BatchNorm2d(256)
        conv3_size = int(self._get_cv_output_size(conv2_size)/2)
        print(f'Con3 size: {conv3_size * 2}')
        print(f'MaxPool3 size: {conv3_size}')

        # linear decision layers
        self.fc1 = nn.Linear(256 * (conv3_size ** 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_of_classes)

    def forward(self, x):
        x1 = F.max_pool2d(self.conv1(x), 2)
        x1 = F.leaky_relu(self.bnorm1(x1))

        x2 = F.max_pool2d(self.conv2(x1), 2)
        x2 = F.leaky_relu(self.bnorm2(x2))

        x3 = F.max_pool2d(self.conv3(x2), 2)
        x3 = F.leaky_relu(self.bnorm3(x3))

        x_flat = x3.view(-1, int(x3.shape.numel() / x3.shape[0]))

        x4 = F.leaky_relu(self.fc1(x_flat))
        x4 = F.dropout(x4, p=0.5, training=self.training)
        x5 = F.leaky_relu(self.fc2(x4))
        x5 = F.dropout(x5, p=0.5, training=self.training)
        x6 = self.fc3(x5)

        return x6  # Return the final output and the feature map from the last conv layer

    # just a handy function to calculate the output of a conv layer
    def _get_cv_output_size(self, input, kernel = kernel_size, stride = 1, padding = 0):
        return int(np.floor((input + 2 * padding - kernel)/stride) + 1)

