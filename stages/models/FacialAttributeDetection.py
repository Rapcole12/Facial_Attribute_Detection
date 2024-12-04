from ..main_stages.MTCNN import MTCNN, save_and_show_image
from ..main_stages.AttributeRecognition import AttributeRecognitionCNN
from ..utils.props import extract_patches
import torch.nn as nn
import numpy as np
import cv2
from ..utils.props import generate_bounding_box_after_pnet, resize_to_square

class FacialAttributeDetection(nn.Module):
    def __init__(self, num_attributes):
        super(FacialAttributeDetection, self).__init__()
        self.mtcnn = MTCNN()
        self.attribute_recognition = AttributeRecognitionCNN(num_attributes)

    def forward(self, x):
        bounding_box_rnet, prob_onet, bounding_box_onet, landmarks_onet = self.mtcnn(x)

        new_bounding_box_onet = generate_bounding_box_after_pnet(bounding_box_onet, prob_onet, .45)

        new_bounding_box_onet = resize_to_square(new_bounding_box_onet)

        new_bounding_box_onet = extract_patches(bounding_box_rnet, new_bounding_box_onet, expected_size=(64, 64))

        # Save and show RNet output
        save_and_show_image(new_bounding_box_onet[0], "onet", 0)

        # image = new_bounding_box_onet[0].cpu().detach().numpy()
        # # Transpose the image to HxWxC format (channels last for OpenCV)
        # image = np.transpose(image, (1, 2, 0))
        # # Normalize the image if necessary (assuming it's between -1 and 1)
        # image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
        # # Display the image using OpenCV
        # cv2.imshow("Tensor Image", image)
        # # Wait for a key press and close the window
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x = self.attribute_recognition(new_bounding_box_onet)
        
        return x, bounding_box_onet, prob_onet
