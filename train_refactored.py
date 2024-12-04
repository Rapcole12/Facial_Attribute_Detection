from stages.models.FacialAttributeDetection_Refactored import FacialAttributeDetection
from data_loading.data_loader_refactored import get_dataloaders
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, Subset, DataLoader
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, loader, criterion, optimizer, epochs=2):

    print("Start Training")
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        for i, data in enumerate(tqdm(loader)):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            attributes, _= model(inputs, device)

            if attributes is None or attributes.size(0) == 0:
                # Skip loss calculation for this batch if no faces were detected
                continue
   
            loss = criterion(attributes, labels)
            loss.backward()
            optimizer.step()
            
            predictions = (torch.sigmoid(attributes) > 0.5).float()
            correct += (predictions == labels).float().sum().item()
            total += labels.numel()

            running_loss += loss.item()

            if i % 250 == 249:    # print every 250 mini-batches
                print(f'[epoch {epoch + 1}, batch {i + 1:5d}] Facial Attribute Detection Loss: {running_loss / 250:.3f}')
                running_loss = 0.0

                print(f'\nAccuracy of the network on last 250 training images: {100 * correct // total} %')
                total = 0
                correct = 0

    print("\nFinished Training")

def evaluation(model, loader):
    # Evaluate accuracy on validation/test set
    correct = 0
    total = 0
    print("\nStart Evaluating")
    with torch.no_grad():
        for data in tqdm(loader):
            images, labels = data

            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            attributes, _= model(images, device)
            if attributes is None:
                # Skip loss calculation for this batch or handle accordingly
                continue
            # apply sigmoid and threshold for binary classification
            predictions = (torch.sigmoid(attributes) > 0.5).float()
            
            # Update correct and total
            correct += (predictions == labels).float().sum().item()  # Count correct predictions
            total += labels.numel()  # Total number of elements in labels (for multi-label tasks)

    # Calculate percentage accuracy
    accuracy = 100 * correct / total
    print(f'\nAccuracy of the network on the test set: {accuracy:.2f} %')


def main():
    img_dir = "celeba_data/celeba-dataset/img_align_celeba/img_align_celeba"
    attr_path = "celeba_data/celeba-dataset/list_attr_celeba.csv/list_attr_celeba.csv"

    selected_features = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
        'Oval_Face', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns', 'Smiling'
    ]

    batch_size = 1

    train_loader, test_loader = get_dataloaders(img_dir, attr_path, selected_features, batch_size=batch_size)

    print(device)

    Facial_Attribute_model = FacialAttributeDetection(len(selected_features)).to(device)

    loss = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(Facial_Attribute_model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 2

    train(Facial_Attribute_model, train_loader, loss, optimizer, epochs=num_epochs)

    evaluation(Facial_Attribute_model, test_loader)

    torch.save(Facial_Attribute_model, 'Face_Attribute_model.pth')


if __name__ == "__main__":
    main()