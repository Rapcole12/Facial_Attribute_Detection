from mtcnn import MTCNN
from ..main_stages.AttributeRecognitionMMCNN import AttributeRecognitionCNN
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np

class FacialAttributeDetection(nn.Module):
    def __init__(self, num_attributes):
        super(FacialAttributeDetection, self).__init__()
        self.mtcnn = MTCNN()
        self.attribute_recognition = AttributeRecognitionCNN(num_attributes)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),  # Resize to the required size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize input
        ])
        self.num_attri = num_attributes

    def forward(self, x, device):
        """
        x: Batch of images (B, C, H, W)
        device: Device to run the computation on
        """
        x = x.cpu().detach().numpy()  # Convert tensor to numpy array
        batch_size = x.shape[0]  # Get batch size
        
        all_attributes = []
        all_bounding_boxes = []

        for i in range(batch_size):
            image = x[i]
            
            cropped_faces = []
            bounding_boxes = []
            detections = self.mtcnn.detect_faces(image)

            for detection in detections:
                x_min, y_min, w, h = detection['box']
                cropped_face = image[y_min:y_min+h, x_min:x_min+w]  # Crop the face
                cropped_faces.append(cropped_face)
                bounding_boxes.append([x_min, y_min, x_min+w, y_min+h])

            # Transform cropped faces for input to the attribute recognition model
            preprocessed_faces = [
                self.transform(face).unsqueeze(0).to(device) for face in cropped_faces
            ]

            if not preprocessed_faces:
                # Append an empty tensor and an empty list if no faces are detected
                all_attributes.append(torch.empty(1, self.num_attri).to(device))
                all_bounding_boxes.append([])
                continue

            # Stack preprocessed faces into a batch
            preprocessed_faces = torch.cat(preprocessed_faces)

            attributes = self.attribute_recognition(preprocessed_faces)

            attributes_avg = attributes.mean(dim=0, keepdim=True)
            
            # Append results to batch-level lists
            all_attributes.append(attributes_avg)
            all_bounding_boxes.append(bounding_boxes)

        # Stack attributes for the entire batch
        all_attributes = torch.cat(all_attributes)

        return all_attributes, all_bounding_boxes
 