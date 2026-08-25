import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_cat_model():
    # 1. ตั้งค่า Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using Device: {device}")

    # 2. Data Transformations (สอดคล้องกับ ImageNet Standard)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # โหลด Dataset
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=0)
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"🐱 Found {num_classes} Cat Breeds: {class_names}")

    # บันทึกชื่อ Class เก็บไว้ให้หน้าเว็บใช้อ่าน
    with open('class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    # 3. โหลด Pretrained MobileNetV3 และปรับ Head Classifier
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Freeze Backbone เริ่มต้น
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    # 4. กำหนด Loss Function และ Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.01)

    # 5. Training Loop
    epochs = 10
    print("\n🚀 Starting Training Pipeline...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(image_datasets['train'])
        train_acc = corrects.double() / len(image_datasets['train'])

        # Validation Check
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(image_datasets['val'])
        val_acc = val_corrects.double() / len(image_datasets['val'])

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}")

    # 6. บันทึก Model Weights
    torch.save(model.state_dict(), 'cat_breed_model.pth')
    print("\n✨ Model training completed & weights saved to 'cat_breed_model.pth'!")

if __name__ == '__main__':
    train_cat_model()