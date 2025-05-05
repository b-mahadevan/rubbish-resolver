# Streamlit Rubbish Classifier  

## 🚀 How It Works  

This Streamlit app uses a pre-trained deep learning model (`waste2.h5`) to classify uploaded images of waste into different categories (e.g., recyclable, organic, hazardous).  

### **Key Features**  
✔ **Image Upload** – Drag and drop or browse to upload waste images  
✔ **Instant Prediction** – Model analyzes the image and returns classification results  
✔ **Probability Display** – See confidence scores for each waste category  

### **How to Run**  
1. Install dependencies from requirements.txt: 
    pip install -r requirements.txt
 
2. Run the app:  
    streamlit run app02.py
  
3. Open `http://localhost:8501` and upload an image!  

### **Requirements**  
- Python 3.8+  
- `waste2.h5` (pre-trained model) in the same directory  

🔍 *For advanced usage (Django backend, deployment), see the full documentation.*