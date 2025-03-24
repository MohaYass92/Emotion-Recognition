# Emotion Recognition using HOG Features and SVM

## 📌 Project Overview  
This project aims to classify human emotions from facial images using **Histogram of Oriented Gradients (HOG)** for feature extraction and **Support Vector Machine (SVM)** for classification. The dataset consists of grayscale facial images categorized into **seven emotion classes**:

😃 **Happy** | 😠 **Angry** | 😢 **Sad** | 😲 **Surprise** | 😐 **Neutral** | 😨 **Fear** | 🫂 **Disgust**  

## 🔍 Features  
- **Preprocessing**: Converts images to grayscale and normalizes them.  
- **Feature Extraction**: Uses **HOG descriptors** to capture facial structure.  
- **Classification**: Implements an **SVM** classifier for emotion recognition.  
- **Dataset Handling**: Uses **Keras ImageDataGenerator** for loading images.  

## 🛠 Tech Stack  
- **Python**  
- **OpenCV**  
- **scikit-image** (for HOG features)  
- **scikit-learn** (for SVM classifier)  
- **TensorFlow/Keras** (for dataset handling)  
- **NumPy & Pandas**  

## 🚀 How to Run  
1. Clone the repository:  
   ```sh
   git clone https://github.com/<your-username>/Emotion-Recognition.git
   cd Emotion-Recognition
   ```
2. Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model:  
   ```sh
   python main.py
   ```

## 📊 Results & Performance  
The model is trained on **28,709 images** and tested on **7,178 images**. Training performance depends on feature extraction time and SVM parameters.  

## 📝 To-Do List  
✅ Improve feature extraction speed  
✅ Optimize SVM parameters  
🔲 Implement CNN for better accuracy  
🔲 Deploy as a web app  

---

