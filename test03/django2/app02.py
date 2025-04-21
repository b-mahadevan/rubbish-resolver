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
        model_path = "D:/project/test03/waste2.h5"
        
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
                'Cardboard': 'Cardboard and paperboard packaging that can be recycled. Flatten boxes and remove any non-paper materials like tape or plastic before recycling.',
                'Food Organics': 'Organic food waste that can be composted. Includes fruit/vegetable scraps, coffee grounds, and eggshells. Avoid meat/dairy in home composting.',
                'Glass': 'Glass bottles and jars that can be recycled indefinitely. Rinse containers and remove lids. Note: Ceramics and window glass cannot go with bottle recycling.',
                'Metal': 'Metal containers (cans, tins) and foil that can be recycled. Rinse food residue. Separate aluminum and steel if required by your local recycling program.',
                'Miscellaneous Trash': 'Non-recyclable waste that must go to landfill. Includes composite materials, heavily soiled items, and certain plastics. Try to minimize this category.',
                'Paper': 'Clean paper products (newspapers, magazines, office paper) that can be recycled. Remove any plastic wrapping or laminated coatings first.',
                'Plastic': 'Plastic containers and packaging marked with recycling symbols ♳♴♵♶♷♸♹. Check local guidelines as only certain types may be recyclable in your area.',
                'Textile Trash': 'Worn-out clothing and fabrics that can be donated or recycled. Even damaged textiles can often be repurposed as rags or insulation material.',
                'Vegetation': 'Yard/garden waste like leaves, grass clippings, and small branches. Can be composted or processed into mulch. Avoid invasive plant species.'
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
    - Cardboard
    - Food Organics
    - Glass
    - Metal
    - Miscellaneous Trash
    - Paper
    - Plastic
    - Textile Trash
    - Vegetation
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