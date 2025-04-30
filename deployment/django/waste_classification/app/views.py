
import os
import logging
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.conf import settings
from PIL import Image
from django import forms

# Configure logging
logger = logging.getLogger(__name__)

class WasteImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload an image of waste',
        help_text='Upload an image to classify waste type'
    )

# Constants
IMAGE_SIZE = 256
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 
    'Textile Trash', 'Vegetation'
]

WASTE_DESCRIPTIONS = {
    'Cardboard': 'Cardboard can be recycled. Break down boxes and remove any non-cardboard materials before recycling.',
    'Food Organics': 'Food waste can be composted. Collect in a compost bin or green waste container.',
    'Glass': 'Glass containers can be recycled. Rinse and remove lids before placing in recycling.',
    'Metal': 'Metal objects can be recycled. Clean and sort different types of metal if possible.',
    'Miscellaneous Trash': 'This is general waste that typically goes to landfill. Try to minimize or find alternative uses.',
    'Paper': 'Paper products can be recycled. Remove any plastic windows or non-paper attachments.',
    'Plastic': 'Plastic items can be recycled. Clean and check local recycling guidelines for specific types.',
    'Textile Trash': 'Textiles can often be donated, recycled, or repurposed. Check local textile recycling options.',
    'Vegetation': 'Vegetation waste can be composted or used in green waste collection.'
}

def load_waste_model():
    try:
        # Precise paths for the model file
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        potential_paths = [
            os.path.join(BASE_DIR, 'app', 'Models', 'waste2.h5'),
            os.path.join(BASE_DIR, 'app', 'waste2.h5'),
            os.path.join(BASE_DIR, 'waste2.h5'),
        ]
        
        model_path = None
        for path in potential_paths:
            logger.info(f"Checking path: {path}")
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            logger.error("No model file found. Searched paths:")
            for path in potential_paths:
                logger.error(path)
            return None
        
        logger.info(f"Loading model from: {model_path}")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Detailed model loading error: {e}")
        import traceback
        traceback.print_exc()
        return None

def index(request):
    # Configure logging to show detailed information
    logging.basicConfig(level=logging.INFO)
    
    # Initialize form
    form = WasteImageUploadForm()
    context = {'form': form}

    # Check if model loading fails
    model = load_waste_model()
    if model is None:
        context['error'] = "Failed to load the machine learning model. Please check model file."
        return render(request, 'index.html', context)

    if request.method == 'POST':
        form = WasteImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Process uploaded image
                image = request.FILES['image']
                img = Image.open(image)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                img_array = img_array / 255.0  # Normalize

                # Make prediction
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = round(100 * np.max(predictions[0]), 2)

                # Prepare prediction probabilities for visualization
                sorted_indices = np.argsort(predictions[0])[::-1]
                top_classes = [CLASS_NAMES[i] for i in sorted_indices[:3]]
                top_probabilities = predictions[0][sorted_indices[:3]] * 100

                context.update({
                    'form': form,
                    'image': image,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'description': WASTE_DESCRIPTIONS.get(predicted_class, "Waste type description not available."),
                    'top_classes': top_classes,
                    'top_probabilities': top_probabilities
                })

            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                import traceback
                traceback.print_exc()
                context['error'] = "An error occurred during image processing. Please try again."

    return render(request, 'index.html', context)
