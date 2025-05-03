import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
N_CLASSES = 9

# Define class names
CLASS_NAMES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Streamlit app configuration
st.set_page_config(page_title="Rubbish Resolver", page_icon=":recycle:", layout="wide")

# Model loading function
@st.cache_resource
def load_waste_model():
    try:
        model_path = "D:/project/test03/streamlit/waste2.h5"
        
        # Check if file exists
        if not os.path.exists(model_path):
            st.error(f"Model file {model_path} not found!")
            return None
        
        # Load model with custom objects and compile
        model = load_model(model_path, compile=False)
        
        # Recompile the model with explicit loss and metrics
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main Streamlit App
def main():
    st.title("🚮 Rubbish Resolver App")

    # Sidebar for navigation
    app_mode = st.sidebar.selectbox("Choose a page", 
        ["Prediction", "About", "Model Information"]
    )

    if app_mode == "Prediction":
        prediction_page()
    elif app_mode == "About":
        about_page()
    else:
        model_info_page()

# Prediction Page
def prediction_page():
    st.header("Rubbish Classification Prediction")
    
    # Load the model
    model = load_waste_model()
    if model is None:
        st.error("Could not load the model. Please check the model file.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of waste", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to classify waste type"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify"):
            try:
                # Preprocess the image
                img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                img_array = img_array / 255.0  # Normalize
                
                # Make prediction
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = round(100 * np.max(predictions[0]), 2)
                
                # Display results
                st.success(f"Prediction: {predicted_class} (Confidence: {confidence}%)")
                
                # Show prediction probabilities
                st.subheader("Prediction Probabilities")
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(CLASS_NAMES))
                ax.barh(y_pos, predictions[0], align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(CLASS_NAMES)
                ax.set_xlabel("Probability")
                ax.set_title("Waste Type Probabilities")
                st.pyplot(fig)
                
                # Waste type description
                
                waste_descriptions = {
                    'Cardboard': 'Corrugated cardboard that should be flattened and recycled with paper products.',
                    'Food Organics': 'Food scraps and biodegradable waste that should be composted or placed in organic waste bins.',
                    'Glass': 'Glass bottles and jars that should be rinsed and recycled (check local guidelines for colors).',
                    'Metal': 'Clean metal cans and foil that can be recycled indefinitely.',
                    'Miscellaneous Trash': 'Non-recyclable waste that should go in general landfill bins.',
                    'Paper': 'Clean paper and cardboard that should be dry and uncontaminated for recycling.',
                    'Plastic': 'Plastic containers, bottles, and packaging; recyclable depending on local guidelines.',
                    'Textile Trash': 'Worn-out fabrics that should be donated if usable or recycled as rags.',
                    'Vegetation': 'Yard trimmings and plant matter that should be composted or processed as green waste.'
                }
                st.info(waste_descriptions.get(predicted_class, "Waste type description not available."))
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# About Page
def about_page():
    st.header("About Waste Classification System")
    st.write("""
    ### Our Mission
    This application helps users identify and properly dispose of different types of waste.
    
    ### How It Works
    1. Upload an image of waste
    2. Our AI model classifies the waste type
    3. Get information about proper disposal
    
    ### Supported Waste Categories
    - Battery
    - Food Waste
    - Brown-Glass
    - Cardboard
    - Clothes
    - Green-Glass
    - Metal
    - Paper
    - Plastic
    """)

# Model Information Page
def model_info_page():
    st.header("Model Information")
    
    # Model details
    st.subheader("Model Specifications")
    st.write(f"Image Size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    st.write(f"Number of Classes: {N_CLASSES}")
    st.write("Supported Waste Categories:")
    for category in CLASS_NAMES:
        st.write(f"- {category}")

# Run the app
if __name__ == "__main__":
    main()