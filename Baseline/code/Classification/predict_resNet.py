import torch
from torchvision import models, transforms
from PIL import Image

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载并预处理图片
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保图片是 RGB 格式
    image = transform(image)  # 应用预处理
    image = image.unsqueeze(0)  # 添加 batch 维度
    return image

# 使用模型进行预测
def predict_image(model, image_path, class_names):
    device = next(model.parameters()).device

    # 预处理图片
    image = preprocess_image(image_path).to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # 返回预测类别
    return class_names[predicted.item()]

if __name__ == '__main__':
    # 定义类别名称
    class_names = ['Pallas', 'Persian', 'Ragdoll', 'Singapura', 'Sphynx']

    # 定义模型结构
    num_classes = len(class_names)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载最佳模型
    model.load_state_dict(torch.load(r'G:\NUS\DLCourse\BaseLine\code\Classification\checkpoints\best_resnet.pth'))
    model.eval()

    # 输入图片路径
    image_path = r'G:\NUS\DLCourse\BaseLine\data\dataset\sample_cats\02.jpg'  # 替换为实际图片路径

    # 进行预测
    predicted_class = predict_image(model, image_path, class_names)
    print(f'Predicted class: {predicted_class}')