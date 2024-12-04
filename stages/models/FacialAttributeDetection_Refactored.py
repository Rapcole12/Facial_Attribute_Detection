from mtcnn import MTCNN
from ..main_stages.AttributeRecognition import AttributeRecognitionCNN
from ..utils.props import extract_patches
import torch.nn as nn
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class FacialAttributeDetection(nn.Module):
    def __init__(self, num_attributes):
        super(FacialAttributeDetection, self).__init__()
        self.mtcnn = MTCNN()
        self.attribute_recognition = AttributeRecognitionCNN(num_attributes)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),  # Resize to the required size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize input
        ])

    def forward(self, x, device):
        x = x.cpu().detach().numpy().squeeze(0)

        cropped_faces = []
        bounding_boxes = []

        detections = self.mtcnn.detect_faces(x)

        for detection in detections:
            x_min, y_min, w, h = detection['box']
            cropped_face = x[y_min:y_min+h, x_min:x_min+w]  # Crop the face
            cropped_faces.append(cropped_face)
            bounding_boxes.append([x_min, y_min, x_min+w, y_min+h])

        # Transform cropped faces for input to the attribute recognition model
        preprocessed_faces = [
            self.transform(face).unsqueeze(0).to(device) for face in cropped_faces
        ]

        if not preprocessed_faces:
            # Return an empty tensor and an empty list if no faces are detected
            return torch.empty(0, self.attribute_recognition.classifier[-1].out_features).to(device), []

        # Stack preprocessed faces into a batch
        preprocessed_faces = torch.cat(preprocessed_faces)

        # Pass the batch of faces through the attribute recognition model
        attributes = self.attribute_recognition(preprocessed_faces)

        attributes_avg = attributes.mean(dim=0, keepdim=True)

        return attributes_avg, bounding_boxes