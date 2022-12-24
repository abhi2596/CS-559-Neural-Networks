# -*- coding: utf-8 -*-
"""HW5_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NqUD33q3IfQjgwocMkjGkyVOV37Zo_w8
"""

import os
import torch
from torchvision import transforms
from torch import nn
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(18432, 128),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(torch.flatten(x, start_dim=1))
        x = self.fc2(x)
        return x


mean, std = (torch.tensor([0.4976, 0.4975, 0.4984]), torch.tensor([0.2877, 0.2891, 0.2883]))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_path = os.path.join(os.pardir, "geometry_dataset\sample_dataset") # change the path to folder where the test file is located
test_dataset = sorted(os.listdir(dataset_path))
model = Net()
model_path = os.path.join(os.getcwd(), "0602-659026651-Chintakunta.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

classes = {0: 'Circle', 1: 'Heptagon', 2: 'Hexagon', 3: 'Nonagon', 4: 'Octagon', 5: 'Pentagon', 6: 'Square', 7: 'Star', 8: 'Triangle'}

for X in test_dataset:
    image = Image.open(os.path.join(dataset_path, X))
    image = transform(image)
    pred = model(image.unsqueeze(dim=0))
    label = classes[pred.argmax().item()]
    print("the predicted output of " + str(X) + " is " + label)

