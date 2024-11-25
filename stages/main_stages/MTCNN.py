from ..mtcnn_stages.pnet import PNet
from ..mtcnn_stages.rnet import RNet
from ..mtcnn_stages.onet import ONet
import torch.nn as nn
import numpy as np
import torch
import torchvision.ops
import matplotlib.pyplot as plt
from ..utils.props import extract_patches

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

        new_bounding_box_pnet = extract_patches(x ,bounding_box_pnet, expected_size=(24, 24))

        prob_rnet, bounding_box_rnet = self.rnet(new_bounding_box_pnet)

        new_bounding_box_rnet = extract_patches(x, bounding_box_rnet, expected_size=(48, 48))
        
        prob_onet, bounding_box_onet, landmarks_onet = self.onet(new_bounding_box_rnet)

        # Return the predictions from the three networks
        return prob_pnet, bounding_box_pnet, prob_rnet, bounding_box_rnet, prob_onet, bounding_box_onet, landmarks_onet
    