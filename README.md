# ♻️ Rubbish Resolver – Waste Classification System

**Rubbish Resolver** is an AI-powered waste classification system designed to promote eco-friendly waste disposal through intelligent image recognition. Using a custom-trained **Convolutional Neural Network (CNN)**, the system classifies waste into **9 distinct categories** and provides real-time predictions through interactive web applications.

---

## 🚀 Project Highlights

### 🧠 Custom CNN Model

- Trained on **8,289 labeled images** representing 9 waste types.
- Built using **TensorFlow** and **Keras** with a custom CNN architecture.

### 🔄 Data Augmentation

- Original dataset: **4,752 images**
- After augmentation: **8,252 images**
- Used **ImageDataGenerator** with rotation, flipping, and zoom techniques.
  
### 📊 Model Performance

| Metric     | Accuracy |
|------------|----------|
| Training   | 87.91%   |
| Validation | 83.76%   |
| Testing    | 80.41%   |

### 📈 Visualization & Analysis

- Employed **Matplotlib** and **NumPy** for:
  - Data exploration
  - Accuracy and loss curve plotting
  - Augmented image previews

### ⚡ Real-Time Prediction Platforms

- **🔹 Streamlit App (Lightweight Demo)**  
  - 🖼️ **Upload Waste Images:** Simple drag-and-drop upload interface.
  - 📈 **Real-Time Inference:** Uses the same CNN model to give instant predictions.
  - 📊 **Prediction Visualization:** Bar chart showing probabilities for all 9 waste classes.
  - 📃 **Descriptive Output:** Each result includes details and proper disposal tips.
  - 🚀 **Quick Deployment:** Lightweight and shareable, ideal for demonstrations or testing.

- **🔹 Rubbish Resolver (Django Web App)**  
  - 📤 **Image Upload:** Upload waste images (JPG, JPEG, PNG) via a clean and responsive interface.
  - 🧠 **Real-Time Prediction:** Classifies uploaded waste items instantly using a custom-trained CNN model.
  - 📊 **Confidence Score Bar:** Visual indicator displays prediction confidence for transparency.
  - 📄 **Predicted Class & Description:** Shows waste category with an eco-friendly disposal description.
  - ♻️ **Disposal Tip:** Offers actionable sustainability guidance based on waste type.
  - 🧩 **Category Overview:** Quick reference boxes for waste types like Recyclable, Organic, and Hazardous.
  - 🌱 **Eco-Friendly Design:** Clean UI that encourages responsible waste handling.
  - ⚙️ **Powered by Django:** Built with Django templates, forms, and view logic integrated with the model.

---

## 📂 Dataset

- **Primary Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)  
- **Alternative Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/luvvalecha/real-waste-dataset)

> 💡 If the UCI link is down, you can use the Kaggle mirror to continue development.

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

| Component             | Tools Used                          |
|----------------------|--------------------------------------|
| Language             | Python                              |
| Deep Learning        | TensorFlow, Keras                   |
| Data Processing      | ImageDataGenerator, NumPy           |
| Visualization        | Matplotlib                          |
| Web Deployment       | Streamlit, Django                   |
| Mobile Optimization  | TensorFlow Lite                     |

---

## 🖼️ Screenshots
- [https://github.com/b-mahadevan/rubbish-resolver/tree/main/deployment/streamlit/demo](https://github.com/b-mahadevan/rubbish-resolver/blob/main/deployment/django/demo/Django.png)
- [https://github.com/b-mahadevan/rubbish-resolver/tree/main/deployment/django/demo](https://github.com/b-mahadevan/rubbish-resolver/blob/main/deployment/streamlit/demo/Streamlit.jpg)

---

## 📱 Future Enhancements

- 🔍 Add image preprocessing (e.g., background removal).
- 📱 Build a mobile app with **TFLite** integration.
- 🌍 Integrate with geolocation APIs to recommend nearby recycling centers.

---
