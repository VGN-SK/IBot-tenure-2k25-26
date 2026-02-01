import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os, shutil, random
from PIL import Image

SOURCE_DIR = "/kaggle/input/microsoft-catsvsdogs-dataset/PetImages"
WORK_DIR   = "/kaggle/working/data"

for split in ["train", "val", "test"]:
    for cls in ["cats", "dogs"]:
        os.makedirs(f"{WORK_DIR}/{split}/{cls}", exist_ok=True)

def is_valid_image(path):
    try:
        Image.open(path).verify()
        return True
    except:
        return False

def split_copy(files, cls):
    valid_files = []

    for f in files:
        path = f"{SOURCE_DIR}/{cls}/{f}"
        if f.lower().endswith(".jpg") and is_valid_image(path):
            valid_files.append(f)

    random.shuffle(valid_files)

    n = len(valid_files)
    t1 = int(0.8 * n)
    t2 = int(0.9 * n)

    for f in valid_files[:t1]:
        shutil.copy(
            f"{SOURCE_DIR}/{cls}/{f}",
            f"{WORK_DIR}/train/{cls.lower()}s/{f}"
        )

    for f in valid_files[t1:t2]:
        shutil.copy(
            f"{SOURCE_DIR}/{cls}/{f}",
            f"{WORK_DIR}/val/{cls.lower()}s/{f}"
        )

    for f in valid_files[t2:]:
        shutil.copy(
            f"{SOURCE_DIR}/{cls}/{f}",
            f"{WORK_DIR}/test/{cls.lower()}s/{f}"
        )

    print(f"{cls}: {len(valid_files)} valid images")

split_copy(os.listdir(f"{SOURCE_DIR}/Cat"), "Cat")
split_copy(os.listdir(f"{SOURCE_DIR}/Dog"), "Dog")

print("✅ Dataset split completed safely")

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

for split in ["train", "val", "test"]:
    for cls in ["cats", "dogs"]:
        path = f"{WORK_DIR}/{split}/{cls}"
        print(split, cls, "→", len(os.listdir(path)))

train_dataset = datasets.ImageFolder("/kaggle/working/data/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder("/kaggle/working/data/val",   transform=val_transforms)
test_dataset  = datasets.ImageFolder("/kaggle/working/data/test",  transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(len(train_dataset), len(val_dataset), len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

num_epochs = 8

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    vloss, vcorrect, vtotal = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            vloss += loss.item()
            _, preds = torch.max(outputs, 1)
            vtotal += labels.size(0)
            vcorrect += (preds == labels).sum().item()

    val_loss = vloss / len(val_loader)
    val_acc  = 100 * vcorrect / vtotal

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

# Unfreeze last residual block
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3}
])

print("Fine-tuning last ResNet block...")

for epoch in range(3):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = criterion(model(imgs), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Fine-tuning epoch {epoch+1} done")
