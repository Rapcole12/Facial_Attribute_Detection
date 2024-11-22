from .pnet import PNet
from .rnet import RNet
from .onet import ONet
from torch.nn import nn
import torch.optim as optim


pnet = PNet()
rnet = RNet()
onet = ONet()

loss_fn = nn.CrossEntropyLoss()

optimizer_p = optim.Adam(pnet.parameters(), lr=0.001)
optimizer_r = optim.Adam(rnet.parameters(), lr=0.001)
optimizer_o = optim.Adam(onet.parameters(), lr=0.001)

num_epochs = 5

