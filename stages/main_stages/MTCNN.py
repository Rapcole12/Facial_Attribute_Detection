from ..mtcnn_stages.pnet import PNet
from ..mtcnn_stages.rnet import RNet
from ..mtcnn_stages.onet import ONet
import torch.nn as nn
import numpy as np
import torch
import torchvision.ops
import matplotlib.pyplot as plt
from ..utils.props import *
import cv2

class MTCNN(nn.Module):
    def __init__(self):
        super(MTCNN, self).__init__()
        # Initialize the individual networks
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, x):
        prob_pnet, bounding_box_pnet = self.pnet(x)

        new_bounding_box_pnet = generate_bounding_box(bounding_box_pnet, prob_pnet, 0.4)

        new_bounding_box_pnet = resize_to_square(new_bounding_box_pnet)

        np.clip(new_bounding_box_pnet, -12, 12)

        new_bounding_box_pnet = extract_patches(x ,new_bounding_box_pnet, expected_size=(24, 24))

        prob_rnet, bounding_box_rnet = self.rnet(new_bounding_box_pnet)

        bboxes_batch = replace_confidence(bounding_box_rnet, prob_rnet)

        new_bounding_box_rnet = adjust_bboxes(bboxes_batch, bounding_box_rnet)

        new_bounding_box_rnet = generate_bounding_box_after_pnet(new_bounding_box_rnet, prob_rnet, .45)

        new_bounding_box_rnet = resize_to_square(new_bounding_box_rnet)

        new_bounding_box_rnet = extract_patches(new_bounding_box_pnet, new_bounding_box_rnet, expected_size=(48, 48))

        prob_onet, bounding_box_onet, landmarks_onet = self.onet(new_bounding_box_rnet)

        bboxes_batch = replace_confidence(bounding_box_onet, prob_onet)

        bounding_box_onet = adjust_bboxes(bboxes_batch, bounding_box_onet)

        # Return the predictions from the three networks
        return new_bounding_box_pnet, prob_onet, bounding_box_onet, landmarks_onet
    