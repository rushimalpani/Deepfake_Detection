import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_label(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    if predictions[0][0] >= 0.5:
        return "Fake"
    else:
        return "Real"

# Mapping of models to their corresponding test sets
model_test_set_mapping = {
    'Models/CudaServer/CudaServer_2k_50_model.h5': 'Test_Dataset/2k dataset',
    'Models/CudaServer/CudaServer_7k_50_model.h5': 'Test_Dataset/7k dataset',
    'Models/CudaServer/CudaServer_20k_50_model.h5': 'Test_Dataset/20k dataset'
}

test_conditions = ['High', 'Low', 'Medium']

for model_name, test_set_base in model_test_set_mapping.items():
    loaded_model = load_model(model_name)  # Load the pre-trained model
    for condition in test_conditions:
        true_labels = []
        predicted_labels = []

        test_dir = os.path.join(test_set_base)  # Adjust the path to the current condition

        # Iterate through subdirectories in the current test image directory
        for label in ['Real', 'Fake']:  # Directly iterate over 'Real' and 'Fake' labels
            label_dir = os.path.join(test_dir, label, condition)
            for filename in os.listdir(label_dir):
                image_path = os.path.join(label_dir, filename)
                true_labels.append(label)  # Assign true label based on subdirectory name
                predicted_label = predict_label(loaded_model, image_path)
                predicted_labels.append(predicted_label)

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=['Real', 'Fake'])

        # Plot confusion matrix as heatmap
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)  # Adjust font size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix Heatmap for {model_name} on {condition}')

        # Save the figure as a PDF file
        plt.savefig(f'{model_name}_{condition}_confusion_matrix.pdf', format='pdf')

        plt.close()  # Close the figure to free up memory