import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Parameters & Hyperparameters
BATCH_SIZE = 16
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 100
NUM_CLASSES = 5

CLASS_NAMES = ['Microcar', 'Pickup Trucks', 'Sedan', 'Sports', 'Van']

DATA_DIR = 'Data'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="training",
    seed=123,
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="validation",
    seed=123,
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# CNN architecture
model = models.Sequential([
    data_augmentation,
    
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Starting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

model.save('car_type_model.h5')
print("Model saved as 'car_type_model.h5'")

# ==========================================
# 5. RESULTS & EVALUATION METRICS
# ==========================================
print("\nEvaluating Model on Test Data...")

y_true_labels = []
y_pred_labels = []

# Loop through the test dataset batch by batch
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0) # Predict the current batch
    y_true_labels.extend(np.argmax(labels.numpy(), axis=1)) # Save actual answers
    y_pred_labels.extend(np.argmax(preds, axis=1))          # Save AI predictions

# Convert to arrays for the matrix
y_true_labels = np.array(y_true_labels)
y_pred_labels = np.array(y_pred_labels)

# Print Accuracy, Precision, Recall, F1-Score
print("\n--- PERFORMANCE METRICS ---")
print(classification_report(
    y_true_labels, 
    y_pred_labels, 
    target_names=CLASS_NAMES, 
    labels=range(NUM_CLASSES),
    zero_division=0
))

# Confusion Matrix Visualization
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(
    y_true_labels, 
    y_pred_labels, 
    labels=range(NUM_CLASSES)
)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix for Car Type Classification")
plt.tight_layout()
plt.show()