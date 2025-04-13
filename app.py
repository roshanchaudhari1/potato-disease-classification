import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Title and instructions
st.title("🥔 Potato Leaf Disease Classification")
st.write("Upload an image of a potato leaf, and the model will classify it.")

# Define class names used during training
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load the model
MODEL_PATH = r"C:/Users/chaud/alo disease classification/potato_disease_model1.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (same as during training!)
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # Show results
    st.markdown(f"### 🔍 Prediction: `{predicted_class}`")
    st.markdown(f"### ✅ Confidence: `{confidence}%`")

    # Show all class probabilities
    st.subheader("📊 Model Confidence Scores:")
    scores = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.bar_chart(scores)
