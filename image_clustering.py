# clustering_model.py
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import cv2


def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)
    return images


def preprocess_images(images, target_size=(100, 100)):
    preprocessed_images = []
    for img in images:
        # Resize the images to a consistent size
        resized_img = cv2.resize(
            img, target_size, interpolation=cv2.INTER_AREA)
        # Preprocess images (you may add more preprocessing steps if needed)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(gray_img)
    return preprocessed_images


def extract_features(preprocessed_images):
    features = [img.flatten() for img in preprocessed_images]
    # Ensure all images have the same shape by stacking them vertically
    features = np.vstack(features)
    return features


def perform_clustering(features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(features)
    return kmeans


def main():
    # Configuration
    num_clusters = 10
    images_folder = "images"
    model_save_path = "clustering_model.pkl"

    # Step 1: Load images
    images = load_images(images_folder)

    # Step 2: Preprocess images
    preprocessed_images = preprocess_images(images)

    # Step 3: Extract features (flatten images)
    features = extract_features(preprocessed_images)

    # Step 4: Perform clustering
    kmeans_model = perform_clustering(features, num_clusters)

    # Save the clustering model for future use
    joblib.dump(kmeans_model, model_save_path)


if __name__ == "__main__":
    main()
