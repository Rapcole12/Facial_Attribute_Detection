from .pnet import PNet
from .rnet import RNet
from .onet import ONet
from data_loader import get_dataloaders
from torch.nn import nn
import torch.optim as optim

img_dir = "celeba_data/celeba-dataset/img_align_celeba/img_align_celeba"
attr_path = "celeba_data/celeba-dataset/list_attr_celeba.csv"

selected_features = [   # 20 features
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
    'Oval_Face', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns', 'Smiling'
]

batch_size = 32  #unsure what we should have for batch size

train_loader, test_loader = get_dataloaders(img_dir, attr_path, selected_features, batch_size=batch_size)


pnet = PNet()
rnet = RNet()
onet = ONet()

loss_fn = nn.CrossEntropyLoss()

optimizer_p = optim.Adam(pnet.parameters(), lr=0.001)
optimizer_r = optim.Adam(rnet.parameters(), lr=0.001)
optimizer_o = optim.Adam(onet.parameters(), lr=0.001)

num_epochs = 5

