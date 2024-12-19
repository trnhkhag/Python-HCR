import cv2
import numpy as np
import pandas as pd
from keras import models

# Load the trained model
model = models.load_model(r'models/improved_model.h5')

# Label dictionary
label_dictionary = {i: chr(i + ord('A')) for i in range(26)}  # Example for A-Z

def preprocess_and_segment(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Extract character images
    char_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        char = thresh[y:y+h, x:x+w]
        char = cv2.resize(char, (28, 28))  # Resize to match model input size
        char = char / 255.0  # Normalize pixel values
        char = np.expand_dims(char, axis=-1)  # Add channel dimension
        char_images.append(char)

    return char_images

# Test the function
image_path = 'test/E1.png'
char_images = preprocess_and_segment(image_path)
cv2.imshow("Char image", char_images[0])
cv2.waitKey(0)

# Predict each character
predicted_text = ""

for char_image in char_images:
    char_image = np.expand_dims(char_image, axis=0)  # Add batch dimension
    prediction = model.predict(char_image)
    predicted_label = np.argmax(prediction)  # Get the index of the highest probability
    predicted_text = predicted_label

print("Predicted result:", predicted_text)