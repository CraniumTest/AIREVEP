import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Function to preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize pixel values
    return image

# Load dataset
def load_data(image_dir, labels_csv):
    labels = pd.read_csv(labels_csv)
    image_paths = [os.path.join(image_dir, f"{idx}.jpg") for idx in labels['ID']]
    images = np.array([load_and_preprocess_image(path) for path in image_paths])
    prices = labels['Price'].values
    return images, prices

# Define the CNN model
def create_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)  # Output layer to predict property price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load data
image_dir = './images'
labels_csv = './property_prices.csv'

images, prices = load_data(image_dir, labels_csv)
X_train, X_val, y_train, y_val = train_test_split(images, prices, test_size=0.2, random_state=42)

# Create and train model
model = create_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save model
model.save('property_valuation_model.h5')
