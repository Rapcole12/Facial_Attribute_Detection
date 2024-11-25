from .main_stages.MTCNN import MTCNN
from .main_stages.AttributeRecognition import AttributeRecognitionCNN
from .utils.props import extract_patches
import torch.nn as nn

class FacialAttributeDetection(nn.Module):
    def __init__(self, num_attributes):
        super(FacialAttributeDetection, self).__init__()
        self.mtcnn = MTCNN()
        self.attribute_recognition = AttributeRecognitionCNN(num_attributes)

    def forward(self, x):
        prob_pnet, bounding_box_pnet, prob_rnet, bounding_box_rnet, prob_onet, bounding_box_onet, landmarks_onet = self.mtcnn(x)

        new_bounding_box_onet = extract_patches(x, bounding_box_onet, expected_size=(64, 64))

        x = self.attribute_recognition(new_bounding_box_onet)
        
        return x