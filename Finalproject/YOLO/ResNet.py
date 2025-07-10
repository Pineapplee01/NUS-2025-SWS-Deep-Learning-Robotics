import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import os



# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# 数据集路径
train_dir = r'Finalproject\Algorithm\dataset\train'
val_dir = r'Finalproject\Algorithm\dataset\validation'
test_dir = r'Finalproject\Algorithm\dataset\test'

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# 加载预训练ResNet并修改最后一层
num_classes = 5
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # 冻结特征提取层

model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换全连接层
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# 训练函数
def train_model(model, train_loader, val_loader, epochs=20):
    device = next(model.parameters()).device
    best_acc = 0

    #创建保存目录
    save_dir = r'Finalproject\Algorithm\YOLO\checkpoints\ResNet'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f'ResNet_epoch_{epoch+1}.pth'))


        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet.pth')

if __name__ == '__main__':
    #GPU加速
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用GPU加速")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    train_model(model, train_loader, val_loader, epochs=20)