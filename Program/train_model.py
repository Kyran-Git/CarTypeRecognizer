import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. PARAMETERS & HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 16
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 40
NUM_CLASSES = 5

CLASS_NAMES = ['Microcar', 'Pickup Trucks', 'Sedan', 'Sports', 'Van']

DATA_DIR = 'Data' 

# ==========================================
# 2. DATA LOADING & AUTOMATIC SPLITTING
# ==========================================
print("Loading and Splitting Data...")

# 70% for Training
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

# 30% for Testing/Validation (Shuffle enabled for fair splitting)
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

# Data Augmentation & Normalization
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),          # Normalization (scaling pixels to 0-1)
    layers.RandomFlip("horizontal"),   # Augmentation
    layers.RandomRotation(0.1),        # Augmentation
    layers.RandomZoom(0.1),            # Augmentation
])

# ==========================================
# 3. ENHANCED BASIC CNN ARCHITECTURE
# ==========================================
model = models.Sequential([
    data_augmentation,
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Block 4 (Extra depth to make the AI smarter)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    # Flattening and Fully Connected Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("Starting Model Training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# Save the trained model 
model.save('car_type_model.h5')
print("Model saved as 'car_type_model.h5'")

# ==========================================
# 5. RESULTS & EVALUATION METRICS
# ==========================================
print("\nEvaluating Model on Test Data...")

y_true_labels = []
y_pred_labels = []

# Loop through the test dataset batch by batch to guarantee perfectly matched labels
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true_labels.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_labels.extend(np.argmax(preds, axis=1))

y_true_labels = np.array(y_true_labels)
y_pred_labels = np.array(y_pred_labels)

# Print Accuracy, Precision, Recall, F1-Score
print("\n--- PERFORMANCE METRICS ---")
print(classification_report(
    y_true_labels, 
    y_pred_labels, 
    target_names=CLASS_NAMES, 
    labels=range(NUM_CLASSES),
    zero_division=0 # Hides the divide-by-zero warning if a folder is unbalanced
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