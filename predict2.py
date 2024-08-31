import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Paths and Parameters
dataset_path = 'C:\\Users\\vivek\\Desktop\\archive\\train'
img_height, img_width = 180, 180

# Recreate the Data Generator to Get Class Labels
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)

# Get the Class Labels
class_labels = list(train_generator.class_indices.keys())

# Load the Trained Model
model = tf.keras.models.load_model('skin_disease_model.h5')

# Function to Load and Preprocess an Image
def load_and_preprocess_image(img_path, img_height=180, img_width=180):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Function to Predict and Display the Results
def predict_image(model, img_path, class_labels, true_label=None):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]
    
    # Display the Image and Prediction
    plt.imshow(image.load_img(img_path))
    if true_label is not None:
        plt.title(f'Predicted: {predicted_class_label} ({predictions[0][predicted_class_index] * 100:.2f}%)\nTrue Label: {true_label}')
    else:
        plt.title(f'Predicted: {predicted_class_label} ({predictions[0][predicted_class_index] * 100:.2f}%)')
    plt.axis('off')
    plt.show()

# Test Image Path
test_img_path = 'C:\\Users\\vivek\\Desktop\\archive\\benign-familial-chronic-pemphigus-9.jpg'

# Actual label for the test image (Manually specified)
# Replace 'Actual Class Label' with the true label, e.g., 'melanoma' or any of the class labels
true_label = 'Actual Class Label'

# Predict the Image
predict_image(model, test_img_path, class_labels, true_label) 
