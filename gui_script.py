# gui_script.py
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import joblib
from PIL import Image, ImageTk


def load_clustering_model(model_path):
    return joblib.load(model_path)


def preprocess_image(image_path, target_size=(100, 100)):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    return gray_img.flatten()


def find_cluster(image_features, clustering_model):
    cluster = clustering_model.predict([image_features])
    return cluster[0]


def load_images_from_cluster(cluster, images_folder, clustering_model):
    cluster_images = []
    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        img = cv2.imread(image_path)
        cluster_label = find_cluster(
            preprocess_image(image_path), clustering_model)
        if cluster_label == cluster:
            cluster_images.append(img)
    return cluster_images


def main():
    # Configuration
    model_path = "clustering_model.pkl"
    images_folder = "images"

    # Load the clustering model
    clustering_model = load_clustering_model(model_path)

    # Create the GUI
    root = tk.Tk()
    root.title("Image Clustering GUI")

    def browse_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg")])
        if file_path:
            image_features = preprocess_image(file_path)
            cluster = find_cluster(image_features, clustering_model)

            # Create a new window to display the selected image and cluster images
            cluster_window = tk.Toplevel(root)
            cluster_window.title(f"Cluster {cluster}")

            # Display the input image
            input_image = cv2.imread(file_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(input_image)
            input_image.thumbnail((200, 200))  # Resize the input image
            input_image_tk = ImageTk.PhotoImage(input_image)
            input_image_label = tk.Label(cluster_window, image=input_image_tk)
            input_image_label.image = input_image_tk
            input_image_label.pack()

            # Load and display all images from the same cluster in a grid layout
            cluster_images = load_images_from_cluster(
                cluster, images_folder, clustering_model)
            cluster_frame = tk.Frame(cluster_window)
            cluster_frame.pack()
            for idx, img in enumerate(cluster_images):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.thumbnail((100, 100))  # Resize the cluster images
                img_tk = ImageTk.PhotoImage(img)
                img_label = tk.Label(cluster_frame, image=img_tk)
                img_label.image = img_tk
                img_label.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)

    browse_button = tk.Button(root, text="Browse Image", command=browse_image)
    browse_button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
