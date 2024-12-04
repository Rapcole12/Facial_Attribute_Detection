import torch
import torch.nn as nn


class AttributeRecognitionCNN(nn.Module):
    def __init__(self, num_attributes):
        """
        :param num_attributes: Number of binary attributes to predict (e.g., gender, glasses, beard).
        """
        super(AttributeRecognitionCNN, self).__init__()
        
        # Shared Feature Extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2),  # Conv1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool1
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),  # Conv2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # MaxPool2
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Conv4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 1000, kernel_size=3, stride=1, padding=1),  # Conv5
            nn.BatchNorm2d(1000),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # MaxPool3
        )
        
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Branches for Binary Classification
        self.attribute_branches = nn.ModuleList([
            nn.Linear(1000, 2) for _ in range(num_attributes)
        ])
    
    def forward(self, x):
        # Shared feature extraction
        x = self.feature_extractor(x)
        x = self.gap(x)  # Global Average Pooling
        x = torch.flatten(x, start_dim=1)  # Flatten the output
        
        # Attribute-specific binary predictions
        outputs = [branch(x) for branch in self.attribute_branches]

        logits = torch.cat([out[:, 1].unsqueeze(1) for out in outputs], dim=1)
        
        return logits  # List of binary outputs for each attribute