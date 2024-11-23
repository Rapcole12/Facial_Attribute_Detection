from MTCNN import MTCNN
from data_loader import get_dataloaders
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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 250 == 249:    # print every 2000 mini-batches
                print(f'[epoch {epoch + 1}, batch {i + 1:5d}] MTCNN Loss: {running_loss / 250:.3f}')
                running_loss = 0.0

                print(f'\nAccuracy of the network on last 250 training images: {100 * correct // total} %')
                total = 0
                correct = 0

    print("\nFinished Training")

def main():
    img_dir = "celeba_data/celeba-dataset/img_align_celeba/img_align_celeba"
    attr_path = "celeba_data/celeba-dataset/list_attr_celeba.csv/list_attr_celeba.csv"

    selected_features = [   # 20 features
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald',
        'Bangs', 'Big_Lips', 'Big_Nose', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
        'Oval_Face', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns', 'Smiling'
    ]

    batch_size = 8

    train_loader, test_loader = get_dataloaders(img_dir, attr_path, selected_features, batch_size=batch_size)

    print(device)

    MTCNN_model = MTCNN().to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = optim.SGD(MTCNN_model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 5

    train(MTCNN_model, train_loader, loss, optimizer, epochs=num_epochs)


if __name__ == "__main__":
    main()