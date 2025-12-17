# test.py
import os
# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.base import BaseEstimator, TransformerMixin

def predict(dataFilePath, bestModelPath):

    #make sure paths are valid
    if not os.path.exists(dataFilePath):
        raise FileNotFoundError(f"Data folder path does not exist: {dataFilePath}")
    if not os.path.exists(bestModelPath):
        raise FileNotFoundError(f"Model path does not exist: {bestModelPath}")
    bestModelPath = os.path.abspath(bestModelPath)
    dataFilePath = os.path.abspath(dataFilePath)
    # Load the model

    pipeline = joblib.load(bestModelPath)

    # Get all image paths from folder
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [
        os.path.join(dataFilePath, f)
        for f in os.listdir(dataFilePath)
        if f.lower().endswith(image_extensions)
    ]
    # Function to extract CNN features in batches
    cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    img_size=(224, 224)
    batch_size=32
    features = []
    batch_imgs = []

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32)
        batch_imgs.append(img)

        # Process batch
        if len(batch_imgs) == batch_size or i == len(image_paths) - 1:
            batch = np.array(batch_imgs)
            batch = preprocess_input(batch)
            feats = cnn.predict(batch, verbose=0)
            features.extend(feats)
            batch_imgs = []

    features = np.array(features)
    # Predict using the pipeline
    pred_index = pipeline.predict(features)

    return pred_index

if __name__ == "__main__":
    data_folder = r"D:\Hossam\study\fourth_year\first_semester\Machine Learning\assignment\Material-Stream-Identification-System\test"
    model_path =r"D:\Hossam\study\fourth_year\first_semester\Machine Learning\assignment\Material-Stream-Identification-System\models\svc_pipeline.pkl"
    preds = predict(data_folder, model_path)
    for i, p in enumerate(preds):
        print(f"Image {i+1}: {p}")