import torch 
import torch.nn as nn

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1)
        self.prelu3 = nn.PReLU()
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU()
        self.fc2_1 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        self.fc2_2 = nn.Linear(128, 4)


    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x) 
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.prelu4(x)
        prob = self.softmax(self.fc2_1(x))
        coor = self.fc2_2(x)

        return prob, coor