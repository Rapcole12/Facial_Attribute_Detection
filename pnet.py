import torch 
import torch.nn as nn

class PNET(nn.Module):
    def __init__(self):
        super(PNET, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1), 
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1), 
            nn.PReLU()  # PReLU3
        )
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        prob = torch.sigmoid(self.conv4_1(x))
        box_offsets = self.conv4_2(x) 
        return prob, box_offsets