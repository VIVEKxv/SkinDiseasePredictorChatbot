import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import matplotlib.pyplot as plt
import google.generativeai as ai

# Initialize Google Generative AI (Gemini) API
API_KEY = "AIzaSyCbp44uFpmTAD1Iws_r1nxeWkmHMBVzOr0"
ai.configure(api_key=API_KEY)
model = ai.GenerativeModel(model_name="gemini-pro")
chat = model.start_chat()

# Set the page configuration
st.set_page_config(page_title="Skin Disease Predictor", layout="centered", initial_sidebar_state="auto")

# Define the hero section
def hero_section():
    st.markdown(
        """
        <style>
        .hero-text {
            font-size: 50px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 20px;
        }
        .description-text {
            font-size: 20px;
            color: #34495E;
            text-align: center;
            margin-bottom: 50px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="hero-text">Skin Disease Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="description-text">Upload an image to detect skin diseases with confidence.</div>', unsafe_allow_html=True)

# Display the hero section
hero_section()

# Load the model
model_path = 'skin_disease_model.keras'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path.")
else:
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Load the class labels from the dataset directory
dataset_path = 'C:\\Users\\vivek\\Desktop\\archive\\train'
img_height, img_width = 180, 180

datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical'
)
class_labels = list(train_generator.class_indices.keys())

# Function to load and preprocess the image
def load_and_preprocess_image(img, img_height=180, img_width=180):
    img = img.resize((img_height, img_width))  # Resize to the target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Function to predict the disease from an image
def predict_image(model, img, class_labels):
    img_array = load_and_preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class_label, confidence

# Streamlit App Interface
st.subheader("Upload an Image")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Predict the uploaded image
    if st.button("Predict"):
        label, confidence = predict_image(model, img, class_labels)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Accuracy:** {confidence:.2f}%")
        
        # Add the prediction result to chatbot input
        chat_input = f"The predicted disease is {label}. How can I assist you further?"

        # Send the message to the chatbot and get the response
        response = chat.send_message(chat_input)
        st.write(f"Chatbot: {response.text}")

        # Optionally display the prediction with matplotlib (optional)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{label} ({confidence:.2f}%)')
        st.pyplot(fig)
else:
    st.info("Please upload an image to proceed.")

# Chatbot Interaction Section
st.subheader("Ask the Chatbot")

# Input box for chatbot queries
user_query = st.text_input("Type your question here...")

if user_query:
    try:
        # Send the user query to the chatbot
        chat_response = chat.send_message(user_query)
        st.write(f"Chatbot: {chat_response.text}")
    except Exception as e:
        st.write(f"Sorry, I encountered an error with the chatbot: {e}")

# Footer section
def footer():
    st.markdown(
        """
        <style>
        .footer {
            font-size: 14px;
            color: #95A5A6;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="footer">Powered by SekiroAI Six</div>', unsafe_allow_html=True)

footer()
