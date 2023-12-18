import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.metrics import F1Score  # Import F1Score from TensorFlow Addons
import numpy as np

# Load your deep learning model here (replace with your code)
model = keras.models.load_model('monkey_pox.h5', custom_objects={'F1Score': F1Score})

def main():
    st.title("Monkeypox Image Classifier")
    st.write("Upload an image to check if it contains monkeypox.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        

# Create the LIME explainer object
        
        img = Image.open(uploaded_image)
        img = img.resize((224, 224))  # Resize to match model input size
        img = np.asarray(img) / 255.0  # Normalize pixel values (assuming 0-255 range)
        img = img.reshape(1, 224, 224, 3) 
        output = model.predict(img)
        class_labels = ["Yes", "No"]  # Replace with your class labels
        predicted_class = class_labels[np.argmax(output)]

        
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Prediction:", predicted_class)
        if predicted_class == "Yes":
            # Display a warning message if monkeypox is predicted
            st.warning("Warning: The model predicts that you might have monkeypox. Please consult a medical professional for further evaluation.")
            st.markdown('[Click here to know more about Monkey Pox](https://www.who.int/news-room/fact-sheets/detail/monkeypox)')
            # st.link("Click here to know more about Monkey Pox", "https://www.who.int/news-room/fact-sheets/detail/monkeypox")
        else:
            st.success("You are healthy as a horse")
        # st.write("Probability:", output.item())

if __name__ == "__main__":
    main()
