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
        self.image_files = sorted(os.listdir(img_dir))[:5000]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #Loads images
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        #loads selected attributes
        attrs = self.attributes_df.iloc[idx, 1:].values
        attrs = np.array(attrs, dtype=np.float32)
        attrs = torch.tensor(attrs, dtype=torch.float32)

        #possible transformation if needed
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
    # Load the attributes CSV
    attributes = pd.read_csv(attr_path)

    # Convert -1 to 0 for binary classification
    attributes.iloc[:, 1:] = attributes.iloc[:, 1:].replace(-1, 0)

    # Select the relevant columns (filename + selected features)
    columns_to_keep = ['image_id'] + selected_features
    attributes = attributes[columns_to_keep]

    # Limit to the first 5,000 rows
    available_images = sorted(os.listdir(img_dir))[:5000]
    attributes = attributes[attributes['image_id'].isin(available_images)].reset_index(drop=True)

    # Split the dataset into 80% training and 20% testing
    train_df, test_df = train_test_split(attributes, test_size=0.2, random_state=42)

    print(f"Train DataFrame: {len(train_df)}, Test DataFrame: {len(test_df)}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((12, 12)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Create datasets
    train_dataset = CelebADataset(img_dir, train_df, transform=transform)
    test_dataset = CelebADataset(img_dir, test_df, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
