from .main_stages.MTCNN import MTCNN
from .main_stages.AttributeRecognition import AttributeRecognitionCNN
from .utils.props import extract_patches
import torch.nn as nn
import numpy as np
import cv2
from .utils.props import generate_bounding_box_after_pnet, resize_to_square

class FacialAttributeDetection(nn.Module):
    def __init__(self, num_attributes):
        super(FacialAttributeDetection, self).__init__()
        self.mtcnn = MTCNN()
        self.attribute_recognition = AttributeRecognitionCNN(num_attributes)

    def forward(self, x):
        bounding_box_rnet, prob_onet, bounding_box_onet, landmarks_onet = self.mtcnn(x)

        new_bounding_box_onet = generate_bounding_box_after_pnet(bounding_box_onet, prob_onet, .45, strides=4, cell_size=24)

        bounding_box_onet =  resize_to_square(new_bounding_box_onet)

        new_bounding_box_onet = extract_patches(bounding_box_rnet, bounding_box_onet, expected_size=(64, 64))

        x = self.attribute_recognition(new_bounding_box_onet)
        
        return x, bounding_box_onet, prob_onet