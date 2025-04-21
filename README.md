# ♻️ Rubbish Resolver – Waste Classification System

Rubbish Resolver is an AI-enabled waste classification system that leverages a custom Convolutional Neural Network (CNN) to classify waste into 9 categories from image input. It promotes eco-conscious waste disposal by providing quick and reliable identification through interactive platforms.

---

## 🚀 Project Highlights

- 🧠 **Custom CNN Model:** Trained on **8,289 labeled waste images** spanning 9 categories including plastic, metal, paper, food organics, etc.
- 🔄 **Data Augmentation:** Utilized **TensorFlow’s `ImageDataGenerator`** to generate **3,500+ new samples**, improving dataset balance and boosting model learning.
- 📊 **Performance Accuracy:**
  - ✅ Training: **87.91%**
  - ✅ Validation: **83.76%**
  - ✅ Test: **80.41%**
- ⚡ **Real-Time Prediction:** Delivered via two platforms:
  1. **Streamlit App** – A lightweight, single-page demo for rapid testing.
  2. **Django App** – A full-stack interface with image uploads, prediction visualizations, and eco-friendly disposal tips.

---

## 🗃 Waste Categories

- 🧃 Plastic  
- 📄 Paper  
- 🥫 Metal  
- 🍌 Food Organics  
- 🧻 Cardboard  
- 🧤 Textile Trash  
- 🍃 Vegetation  
- 🗑️ Miscellaneous Trash  
- 🍾 Glass

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


