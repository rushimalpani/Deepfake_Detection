import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
loaded_model = load_model('model_file1.h5')  # Replace with the path to your model

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_label(image_path):
    img_array = preprocess_image(image_path)
    predictions = loaded_model.predict(img_array)
    if predictions[0][0] >= 0.5:
        return "deepfake"
    else:
        return "real"

# Paths to test image directory
test_dir = "test_images"

# List to store true and predicted labels
true_labels = []
predicted_labels = []

# Iterate through subdirectories in test image directory
for label in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label)
    if os.path.isdir(label_dir):
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            true_labels.append(label)  # Assign true label based on subdirectory name
            predicted_label = predict_label(image_path)
            predicted_labels.append(predicted_label)

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Deepfake'], yticklabels=['Real', 'Deepfake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()
