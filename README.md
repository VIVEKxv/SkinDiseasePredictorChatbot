# Skin Disease Predictor with Chatbot

## Overview

This application is a web-based tool that allows users to upload images of skin conditions for disease prediction. It uses a TensorFlow model to predict the disease and provides additional information through a chatbot powered by Google Generative AI (Gemini). The chatbot can answer questions and provide assistance based on the predicted disease.

## Features

- **Image Upload**: Allows users to upload images of skin conditions.
- **Disease Prediction**: Predicts the disease from the uploaded image using a pre-trained TensorFlow model.
- **Chatbot Interaction**: Provides additional assistance through a chatbot based on the predicted disease.
- **User Queries**: Users can interact with the chatbot by asking questions.

## Setup

1. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required Python libraries using pip:

   ```bash
   pip install streamlit tensorflow pillow numpy matplotlib google-generativeai
