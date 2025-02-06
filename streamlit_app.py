import streamlit as st
from predict import predict
from load_class import load_class_names
import tensorflow as tf
import tensorflow_hub as hub
from tf_keras.models import load_model
from PIL import Image
import os
from initiate_model import initialize_model
import numpy as np

def load_model_for_streamlit():
    model_path = 'flower_classifier_model.h5'
    if not os.path.exists(model_path):
        st.warning("Model not found. Training new model...")
        initialize_model()
        st.success("Model training completed!")
    
    custom_objects = {
        'KerasLayer': hub.KerasLayer,
        'keras_layer': hub.KerasLayer
    }
    return load_model(model_path, custom_objects=custom_objects, compile=False)

def main():
    st.set_page_config(
        page_title="Flower Classifier",
        page_icon="🌸",
        layout="wide"
    )
    
    st.title("🌸 Flower Classifier")
    st.write("Upload an image of a flower and let AI identify it!")

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a deep learning model based on MobileNetV2 "
        "to classify 102 different types of flowers with high accuracy."
    )

    # Load model
    try:
        model = load_model_for_streamlit()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Load class names
    try:
        class_names = load_class_names('label_map.json')
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Save temporary file for prediction
            temp_path = "temp_image.jpg"
            image.save(temp_path, format='JPEG', quality=95)
            
            # Make prediction
            with st.spinner('Analyzing image...'):
                probs, classes = predict(temp_path, model, top_k=5)
            
            # Display results in a nice format
            st.subheader("Predictions:")
            
            # Create a data container for results
            results_container = st.container()
            
            with results_container:
                for i in range(len(classes)):
                    col1, col2, col3 = st.columns([2, 1, 4])
                    with col1:
                        st.write(f"**{class_names[str(classes[i])]}**")
                    with col2:
                        st.write(f"{probs[i]*100:.1f}%")
                    with col3:
                        # Create a progress bar
                        st.progress(float(probs[i]))
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()
