import torch 
import torch.nn as nn

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1), 
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Linear(576, 128),
            nn.PReLU(),

        )

        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU()
        self.dense5_1 = nn.Linear(128, 2)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.dense4(x)
        x = self.prelu4(x)
        detector = self.dense5_1(x)
        bbox_reg = self.dense5_2(x)
        return detector, bbox_reg