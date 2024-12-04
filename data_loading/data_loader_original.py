import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class CelebADataset(Dataset):
    def __init__(self, img_dir, attributes_df, transform=None):
        """
        Args:
            img_dir (str): Path to the directory with images.
            attributes_df (pd.DataFrame): DataFrame containing image filenames and selected attributes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.attributes_df = attributes_df
        #self.image_files = sorted(os.listdir(img_dir))[:5000]
        self.image_files = sorted(os.listdir(img_dir))[:len(self.attributes_df)]
        self.transform = transform

    def __len__(self):
        return min(len(self.attributes_df), len(self.image_files))

    def __getitem__(self, idx):
        #loags images
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        #loads selected attributes
        attrs = self.attributes_df.iloc[idx, 1:].values.astype('float32')

        #tensor conversion
        attrs = torch.tensor(attrs, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, attrs


def get_dataloaders(img_dir, attr_path, selected_features, batch_size=32):
    """
    Create DataLoaders for training and testing using an 80/20 split.

    Args:
        img_dir (str): Path to the directory with images.
        attr_path (str): Path to the attributes CSV file.
        selected_features (list): List of 20 features to use from the dataset.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        DataLoader: Training and testing DataLoaders.
    """
    #loads CSV
    attributes = pd.read_csv(attr_path)

    #converts -1 to 0 for binary classification
    attributes.iloc[:, 1:] = attributes.iloc[:, 1:].replace(-1, 0)

    #selects relevant columns from the csv
    columns_to_keep = ['image_id'] + selected_features
    attributes = attributes[columns_to_keep]
    attributes.iloc[:, 1:] = attributes.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Limit to the first 10,000 rows CAN BE CHANGED IF SIZE INCREASED
    attributes = attributes.iloc[:10000]

    #ensure correct number of attributes
    image_files = sorted(os.listdir(img_dir))[:len(attributes)]
    if len(image_files) != len(attributes):
        raise ValueError("Mismatch between number of images and attributes")

    #80 percent training and 20 percent testing
    train_df, test_df = train_test_split(attributes, test_size=0.2, random_state=42)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((12, 12)),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  #normalizes
    ])

    train_dataset = CelebADataset(img_dir, train_df, transform=transform)
    test_dataset = CelebADataset(img_dir, test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader