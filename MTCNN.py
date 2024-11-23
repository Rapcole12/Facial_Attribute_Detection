from pnet import PNet
from rnet import RNet
from onet import ONet
import torch.nn as nn

class MTCNN(nn.Module):
    def __init__(self):
        super(MTCNN, self).__init__()
        # Initialize the individual networks
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, x):
        # PNet: Proposal network (Initial face candidate detection)
        prob_pnet, bounding_box_pnet = self.pnet(x)

        prob_rnet, bounding_box_rnet = self.rnet(bounding_box_pnet)
        
        prob_onet, bounding_box_onet, landmarks_onet = self.onet(bounding_box_rnet)

        # Return the predictions from the three networks
        return prob_pnet, bounding_box_pnet, prob_rnet, bounding_box_rnet, prob_onet, bounding_box_onet, landmarks_onet