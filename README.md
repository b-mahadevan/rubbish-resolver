# ♻️ Rubbish Resolver – Waste Classification System

Rubbish Resolver is an AI-enabled waste classification system that leverages a custom Convolutional Neural Network (CNN) to classify waste into 9 categories from image input. It promotes eco-conscious waste disposal by providing quick and reliable identification through interactive platforms.

---

## 🚀 Project Highlights

- 🧠 Custom CNN Model: Developed using a CNN trained on 8,289 labeled images across 9 waste categories, including plastic, cardboard, metal, and glass.
- 🔄 Data Augmentation: Expanded the dataset from 4,752 to 8,252 images using **ImageDataGenerator** (rotation, flip, zoom), resulting in a 12% improvement in model accuracy.
- 📊 **Performance Accuracy:**
  - ✅ Training: **87.91%**
  - ✅ Validation: **83.76%**
  - ✅ Test: **80.41%**
- 📈 Visualization & Processing: Used Matplotlib and NumPy for data analysis and visualization.
- ⚡ **Real-Time Prediction:** Delivered via two platforms:
  1. **Streamlit App** – A lightweight, single-page demo for rapid testing.
  2. **Django App** – A full-stack interface with image uploads, prediction visualizations, and eco-friendly disposal tips.

---

## 📂 Dataset

The dataset used for training and testing is available at the following sources:

- ✅ **Primary Source (UCI Machine Learning Repository):** [Click here](https://archive.ics.uci.edu/dataset/908/realwaste)
- 🔁 **Alternative Source (Kaggle):** [Click here](https://www.kaggle.com/datasets/luvvalecha/real-waste-dataset)

If the primary link is unavailable, feel free to use the alternative source.

---

## 🗃 Waste Categories

- 📦 Cardboard
- 🥦 Food Organics
- 🍾 Glass  
- 🥫 Metal
- 🗑️ Miscellaneous Trash 
- 📄 Paper 
- 🧴 Plastic
- 🧤 Textile Trash 
- 🍃 Vegetation

---

## 🛠️ Tech Stack

- **Languages & Frameworks:** Python, TensorFlow, Keras
- **Model Type:** Convolutional Neural Network (CNN)
- **Data Processing:** ImageDataGenerator, Data Augmentation
- **Deployment:**
  - Streamlit (for lightweight demo UI)
  - Django (for full-stack web application)
  - TensorFlow Lite (for mobile optimization)
 
---


