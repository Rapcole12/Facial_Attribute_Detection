import torch 
import torch.nn as nn

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),

            nn.Linear(256, 5),
            nn.Sigmoid(),
    
        )

    def forward(self, x):
        return self.pre_layer(x)