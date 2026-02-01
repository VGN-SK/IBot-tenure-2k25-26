import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

import train # Import training history

train_losses = train.train_losses
val_losses   = train.val_losses
train_accs   = train.train_accs
val_accs     = train.val_accs

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_dataset  = datasets.ImageFolder("/kaggle/working/data/test",  transform=val_transforms)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

correct, total = 0, 0
all_preds, all_labels, all_imgs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu())
        all_labels.extend(labels.cpu())
        all_imgs.extend(imgs.cpu())

print(f"Test Accuracy: {100 * correct / total:.2f}%")

correct_idx = [i for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]
wrong_idx   = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]

def show_images(indices, title):
    plt.figure(figsize=(12,5))
    for i, idx in enumerate(indices[:10]):
        img = all_imgs[idx].permute(1,2,0)
        img = img * torch.tensor([0.229,0.224,0.225]) + torch.tensor([0.485,0.456,0.406])
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"P:{all_preds[idx].item()} T:{all_labels[idx].item()}")
    plt.suptitle(title)
    plt.show()

show_images(correct_idx, "Correct Predictions")
show_images(wrong_idx, "Incorrect Predictions")

# Training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.show()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.savefig("data_analysis.png")
