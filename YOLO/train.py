from yolov5 import YOLOv5  # 使用同目录下的 yolov5 模块
import subprocess
import sys
from pathlib import Path
import argparse


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv5 on a custom dataset"
    )
    
    # 默认指向项目 Algorithm/dataset/data.yaml 的绝对路径
    parser.add_argument(
        '--datayaml', '--data', dest='data', type=str,
        default=str(Path(__file__).resolve().parent.parent / 'dataset' / 'data.yaml'),
        help='Path to data.yaml'
    )
    parser.add_argument('--dataset', type=str, default='../dataset', help='Path to dataset')
    parser.add_argument('--weights', type=str, default='../../yolov5-master/yolov5s.pt', help='Path to weights file')
    
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--device', type=str, default='0', help='Device to use for training (e.g., "0" for GPU 0, "cpu" for CPU)')
    
    
    parser.add_argument('--project', type=str, default='yolov5_finetuned', help='Project name for training results')
    parser.add_argument('--name', type=str, default='yolov5', help='Name of the training experiment')

    return parser.parse_args()
    
def main():
    args = parse_args()

    # 训练数据集路径
    dataset_path = Path(args.dataset)
    # 创建YOLOv5对象
    yolo = YOLOv5()
    # 开始训练
    yolo.finetune(
        data_yaml   = args.data,
        weights     = args.weights,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        img_size    = args.img_size,
        lr          = args.lr,
        project     = args.project,
        name        = args.name,
    )

if __name__ == '__main__':
    main()