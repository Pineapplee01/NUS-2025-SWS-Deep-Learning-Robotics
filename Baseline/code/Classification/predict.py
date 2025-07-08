from PIL import Image
import torch
import torchvision.transforms as T

# 1. 用YOLOv7检测
yolo_model = torch.hub.load(
    'code/yolov7', 
    'custom', 
    path_or_model='yolov7.pt', 
    ource='local'
)

results = yolo_model('your_image.jpg')
boxes = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

# 2. 对每个检测框裁剪并分类
img = Image.open('your_image.jpg').convert('RGB')
transform = T.Compose([
    T.Resize((256, 128)),  # 与你的分类模型输入一致
    T.ToTensor(),
])

for box in boxes:
    x1, y1, x2, y2 = map(int, box[:4])
    crop = img.crop((x1, y1, x2, y2))
    input_tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        pred = classifier(input_tensor)  # classifier为你的五分类模型
        pred_class = pred.argmax(dim=1).item()
    print(f"检测框: {x1, y1, x2, y2} 分类结果: {pred_class}")