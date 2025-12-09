import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from joblib import Parallel, delayed

dataset_path = "dataset/raw"
augmented_path = "dataset/augmented"
classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
img_size = (128, 128)  

for cls in classes:
    Path(os.path.join(augmented_path, cls)).mkdir(parents=True, exist_ok=True)


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
])


def process_image(img_path, output_folder, aug_times):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return

    img = cv2.resize(img, img_size)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = img_gray / 255.0

    save_path = os.path.join(output_folder, f"{Path(img_name).stem}_proc.png")
    cv2.imwrite(save_path, (img_norm*255).astype(np.uint8))

    for i in range(aug_times):
        augmented = transform(image=img_gray)
        aug_img = augmented["image"]
        aug_save_path = os.path.join(output_folder, f"{Path(img_name).stem}_aug{i}.png")
        cv2.imwrite(aug_save_path, aug_img)


for cls in classes:
    input_folder = os.path.join(dataset_path, cls)
    output_folder = os.path.join(augmented_path, cls)
    images = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
    n_original = len(images)
    n_target = int(np.ceil(n_original * 1.3))
    n_aug_needed = n_target - n_original
    aug_times = int(np.ceil(n_aug_needed / n_original))

    print(f"Processing class: {cls} ({n_original} images, {aug_times} augmentations per image)")

    Parallel(n_jobs=-1)(
        delayed(process_image)(img_path, output_folder, aug_times) for img_path in tqdm(images)
    )

all_images = []
all_labels = []

for cls in classes:
    folder = os.path.join(augmented_path, cls)
    for img_name in os.listdir(folder):
        all_images.append(os.path.join(folder, img_name))
        all_labels.append(cls)

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

print(f"Train images: {len(train_imgs)}, Validation images: {len(val_imgs)}")
