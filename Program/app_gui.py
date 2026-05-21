import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Load the trained model
try:
    model = tf.keras.models.load_model('car_type_model.h5')
except:
    print("Error: Model not found. Please run train_model.py first!")
    exit()

CLASS_NAMES = ['Microcar', 'Pickup Trucks', 'Sedan', 'Sports', 'Van']
IMG_SIZE = 150

def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    # Display the selected image on the GUI
    img = Image.open(file_path)
    img_display = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_display)
    image_label.configure(image=img_tk)
    image_label.image = img_tk
    
    # Preprocess image for the model
    # Convert image to RGB to avoid errors with PNGs that have an alpha channel
    img_array = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Update result label
    result_label.config(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")

# GUI Setup
root = tk.Tk()
root.title("Car Type Recognizer")
root.geometry("450x550")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Car Type Image Classifier", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

upload_btn = tk.Button(root, text="Upload Image", command=classify_image, font=("Helvetica", 12), bg="#4CAF50", fg="white")
upload_btn.pack(pady=10)

image_label = tk.Label(root, bg="#f0f0f0")
image_label.pack(pady=10)

result_label = tk.Label(root, text="Upload an image to see the prediction.", font=("Helvetica", 14), bg="#f0f0f0", fg="#333333")
result_label.pack(pady=20)

root.mainloop()