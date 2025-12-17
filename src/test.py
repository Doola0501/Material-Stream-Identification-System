# test.py
import os
# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.base import BaseEstimator, TransformerMixin

# Custom CNN Feature Extractor
class CNNFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.cnn = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    def fit(self, X, y=None):
        print("CNN FeatureExtractor ready.")
        return self

    def transform(self, X):
        features = []
        batch_imgs = []

        for i, img_path in enumerate(X):
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32)
            batch_imgs.append(img)

            if len(batch_imgs) == self.batch_size or i == len(X) - 1:
                batch = np.array(batch_imgs)
                batch = preprocess_input(batch)
                feats = self.cnn.predict(batch, verbose=0)
                features.extend(feats)
                batch_imgs = []

        return np.array(features)


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

    # Predict using the pipeline
    pred_index = pipeline.predict(image_paths)

    return pred_index

if __name__ == "__main__":
    data_folder = r"D:\Hossam\study\fourth_year\first_semester\Machine Learning\assignment\Material-Stream-Identification-System\test"
    model_path =r"D:\Hossam\study\fourth_year\first_semester\Machine Learning\assignment\Material-Stream-Identification-System\models\svc_pipeline.pkl"
    preds = predict(data_folder, model_path)
    for i, p in enumerate(preds):
        print(f"Image {i+1}: {p}")