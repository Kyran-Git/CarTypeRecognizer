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
EPOCHS = 15
NUM_CLASSES = 2

CLASS_NAMES = ['Microcar', 'Pickup Trucks'] #, 'Sedan', 'Sports', 'Van'

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
    label_mode='categorical',
    shuffle=False
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

# Results and evaluation metrics
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_true_labels = np.argmax(y_true, axis=1)

y_pred = model.predict(test_dataset)
y_pred_labels = np.argmax(y_pred, axis=1)

# Print accuracy, precision, recall, and F1-score
print("\n--- PERFORMANCE METRICS ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=CLASS_NAMES))

# Confusion Matrix Visualization
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix for Car Type Classification")
plt.tight_layout()
plt.show()