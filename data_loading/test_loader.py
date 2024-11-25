import os
import pandas as pd
from data_loading.data_loader import CelebADataset
from torchvision import transforms
import matplotlib.pyplot as plt


img_dir = "celeba_data/celeba-dataset/img_align_celeba/img_align_celeba"
attr_path = "celeba_data/celeba-dataset/list_attr_celeba.csv/list_attr_celeba.csv"


selected_features = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
    'Oval_Face', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns', 'Smiling'
]


attributes = pd.read_csv(attr_path)
attributes.iloc[:, 1:] = attributes.iloc[:, 1:].replace(-1, 0)  # Convert -1 to 0
columns_to_keep = ['image_id'] + selected_features
attributes = attributes[columns_to_keep]
attributes = attributes.iloc[:5000]


attributes.iloc[:, 1:] = attributes.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


dataset = CelebADataset(img_dir, attributes, transform=transform)


def test_dataset(dataset, num_samples=5):
    print(f"Testing dataset with {len(dataset)} samples...")

    for idx in range(num_samples):
        try:
            # Retrieve the image and attributes
            image, attrs = dataset[idx]
            image = image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC for display

            # Check alignment with the CSV file
            filename = attributes.iloc[idx, 0]
            print(f"\nSample {idx + 1}:")
            print(f"Image Shape: {image.shape}")
            print(f"Attributes (Tensor): {attrs.numpy()}")
            print(f"Filename from CSV: {filename}")

            # Display the image
            plt.imshow((image * 0.5 + 0.5))  # Unnormalize the image for visualization
            plt.title(f"Attributes for {filename}: {attrs.numpy()}")
            plt.axis("off")
            plt.show()
        except IndexError as e:
            print(f"IndexError: {e}")
            break


# Run the test
test_dataset(dataset)