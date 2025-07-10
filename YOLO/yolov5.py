import torch
import subprocess

class YOLOv5:
    def __init__(self,
                 model_name: str = 'yolov5s',  # 可选 yolov5s, yolov5m, yolov5l, yolov5x
                 device: str = ''):            # '' 自动选 GPU/CPU, 或 'cpu','0','0,1'
        # 通过 torch.hub 加载 ultralytics/yolov5 预训练模型
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        # 如果 device 为空，则选择 GPU 索引 '0' 或 'cpu'
        
        if not device:
            device = '0' if torch.cuda.is_available() else 'cpu'
        # 转换为 torch 可识别的设备字符串
        if device.isdigit():
            torch_device = f'cuda:{device}'
        else:
            torch_device = device
        self.model.to(torch_device)
        # 保存原始设备格式，用于训练脚本参数
        self.device = device

    def predict(self,
                imgs,                   # 单张图片路径/ndarray 或 多张图片列表
                size: int = 640,        # 推理输入尺寸
                conf_thres: float = 0.25,
                iou_thres: float = 0.45,
                classes: list[int] = None,
                agnostic_nms: bool = False,
                augment: bool = False):
        """
        返回结果封装：
          results = self.model(imgs, size=size, conf=conf_thres, iou=iou_thres,
                               classes=classes, agnostic_nms=agnostic_nms, augment=augment)
          results.xyxy  # list of tensors [[x1,y1,x2,y2,conf,cls], …]
          results.pandas().xyxy[0]  # pandas.DataFrame
        """
        results = self.model(imgs,
                             size=size,
                             conf=conf_thres,
                             iou=iou_thres,
                             classes=classes,
                             agnostic_nms=agnostic_nms,
                             augment=augment)
        return results

    def finetune(self,
                 data_yaml: str,         # 自定义数据集配置文件 .yaml
                 weights: str = 'yolov5s.pt',
                 epochs: int = 50,
                 batch_size: int = 16,
                 img_size: int = 640,
                 lr: float = 0.01,
                 project: str = 'runs/train', # 保存结果的项目文件夹
                 name: str = 'exp'):       # 实验名称
        """
        通过 Subprocess 调用 train.py 进行微调
        """
        cmd = [
            'python', 'train.py',
            '--imgsz', str(img_size),      # input image size
            '--batch-size', str(batch_size),# batch size
            '--epochs', str(epochs),        # number of epochs
            '--data', data_yaml,
            '--weights', weights,
            '--device', self.device,
            '--project', project,
            '--name', name
        ]
        subprocess.run(cmd, cwd=r'g:\NUS\DLCourse\Finalproject\yolov5-master')