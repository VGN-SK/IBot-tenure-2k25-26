import torch
import Simple_CNN_architecture as CNN_Model
from CNN_training import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cpu")  # safest on Windows

model = CNN_Model.SimpleNN(no_of_classes=10)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = CustomDataset("test.csv", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")