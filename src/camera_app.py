import os
# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2
import joblib

#CNN Feature Extractor
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

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svc_pipeline.pkl")
svc_pipeline_2 = joblib.load(MODEL_PATH)

#  Classes Mapping 
classes = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "unknown"]

#  Open Camera 
cap = cv2.VideoCapture(0)  # Always camera index 0
if not cap.isOpened():
    raise RuntimeError("Cannot open camera 0")

# Real-Time Prediction
frame_count = 0
TEMP_IMG_PATH = "temp_frame.jpg"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process one frame every 30 frames
    if frame_count % 30 == 0:
        frame_resized = cv2.resize(frame, (224, 224))
        cv2.imwrite(TEMP_IMG_PATH, frame_resized)

        predicted_index = svc_pipeline_2.predict([TEMP_IMG_PATH])[0]
        predicted_label = classes[int(predicted_index)]  # Map numeric index to class name

        # Print prediction to terminal
        print(f"Frame {frame_count}: Predicted label -> {predicted_label}")
