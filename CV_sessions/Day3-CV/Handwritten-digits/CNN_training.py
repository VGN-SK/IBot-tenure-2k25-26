import Simple_CNN_architecture as CNN_Model
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset) :
    def __init__(self, csv_path, transform=None) :

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        file_path = os.path.join(data_dir, csv_path)

        self.data_frame = pd.read_csv(file_path)
        print("Loading CSV from:", file_path)
        self.transform = transform

    def __len__(self) :
        return len(self.data_frame)
    
    def __getitem__(self, idx) :
        image = self.data_frame.iloc[idx, 1:].values.reshape(28, 28).astype(np.uint8)
        label = torch.tensor(self.data_frame.iloc[idx, 0],dtype=torch.long)
        image = Image.fromarray(image)

        if self.transform :
            image = self.transform(image)

        return image, label
    
def main():
    transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)) ])

    train_dataset = CustomDataset(csv_path='train.csv', transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    test_dataset = CustomDataset(csv_path='test.csv', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model = CNN_Model.SimpleNN(no_of_classes=10)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Train dataset length:", len(train_dataset))
    print("Train loader length:", len(train_loader))

    num_epochs = 3

    print("Starting Training...")

    for epoch in range(num_epochs) :
        print("Entered epoch loop", epoch)
        model.train()
        running_loss = 0.0

        for images, labels in train_loader :
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), "model.pth")

    print("Training Complete!")

if __name__ == "__main__":
    main()