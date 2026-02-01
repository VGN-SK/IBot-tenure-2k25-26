import torch
import Simple_CNN_architecture as CNN_Model

from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class TestImages(Dataset) :
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
        image = self.data_frame.iloc[idx].values.reshape(28, 28).astype(np.uint8)
        image = Image.fromarray(image)

        if self.transform :
            image = self.transform(image)

        return image


device = torch.device("cpu")

model = CNN_Model.SimpleNN(no_of_classes=10)
model.load_state_dict(torch.load("model.pth", map_location=device ))
model = model.to(device)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
model.eval()

no_of_images = 10
start = 130

test_image = TestImages("test.csv", transform= transform )

fig , axes = plt.subplots(no_of_images//5,5,figsize =(20,4*no_of_images//5))
axes = axes.ravel()
for i in range(start,no_of_images+start):
    image = test_image[i].unsqueeze(0).to(device)
    image_numpy = image.squeeze().cpu().numpy()

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    axes[i-start].imshow(image_numpy, cmap='gray')
    axes[i-start].set_title(f"Image {i+1} - Predicted: {pred}")
    axes[i-start].axis('off')
plt.show()