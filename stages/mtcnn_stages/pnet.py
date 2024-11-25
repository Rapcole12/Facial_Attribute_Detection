import torch 
import torch.nn as nn

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.prelu3 = nn.PReLU()
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        prob = self.softmax(self.conv4_1(x))
        bounding_box = self.conv4_2(x)
        return prob, bounding_box
