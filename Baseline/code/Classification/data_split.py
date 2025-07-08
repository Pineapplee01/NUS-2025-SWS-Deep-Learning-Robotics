import os
import shutil
import random

# 原始数据集路径，每个类别一个文件夹
src_root = r'G:\NUS\DLCourse\BaseLine\data\dataset'
dst_root = r'G:\NUS\DLCourse\BaseLine\data\dataset\images'
cat_names = ['Pallas','Persian','Ragdoll','Singapura','Sphynx']

train_ratio = 0.85  # 8.5:1.5

random.seed(42)  # 保证可复现

for cat_name in cat_names:
    class_dir = os.path.join(src_root, cat_name)
    if not os.path.isdir(class_dir):
        continue

    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    # 目标文件夹
    train_class_dir = os.path.join(dst_root, 'train', cat_name)
    val_class_dir = os.path.join(dst_root, 'validation', cat_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # 拷贝训练集
    for img in train_imgs:
        src_path = os.path.join(class_dir, img)
        dst_path = os.path.join(train_class_dir, img)
        shutil.copy2(src_path, dst_path)
    # 拷贝验证集
    for img in val_imgs:
        src_path = os.path.join(class_dir, img)
        dst_path = os.path.join(val_class_dir, img)
        shutil.copy2(src_path, dst_path)

print("数据集划分完成！")