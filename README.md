# 🧠 Chest X-Ray Classification using CNN (TensorFlow)

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify chest X-ray images (e.g., Pneumonia vs Normal).

The pipeline includes:
- Data preprocessing with augmentation  
- CNN model architecture  
- Training with callbacks  
- Model saving & loading  
- Prediction on single grayscale images  

---

## 🚀 Features
- Image augmentation using `ImageDataGenerator`
- Deep CNN with:
  - Conv2D + BatchNormalization
  - MaxPooling layers
- Performance metrics:
  - Accuracy
  - Recall (important for medical diagnosis)
- Early stopping & learning rate scheduling
- Supports **grayscale image prediction**
- End-to-end pipeline (train → save → predict)

---

## 🗂️ Project Structure
```
├── train/
├── val/
├── cnn_model.h5
├── class_map.json
├── notebook.ipynb
└── README.md
```

---

## ⚙️ Installation
```
pip install tensorflow numpy matplotlib
```

---

## 📊 Data Preparation
```
train/
   ├── NORMAL/
   ├── PNEUMONIA/

val/
   ├── NORMAL/
   ├── PNEUMONIA/
```

---

## 🧪 Model Architecture
- Input: 224 x 224 x 3
- Conv2D → BatchNorm → MaxPool layers
- Fully connected Dense layers

---

## 🏋️ Training
```python
model.fit(train_gen, validation_data=val_gen, epochs=15)
```

---

## 💾 Save Model
```python
model.save("cnn_model.h5")
```

---

## 🔄 Load Model
```python
model = tf.keras.models.load_model("cnn_model.h5")
```

---

## 🔍 Prediction
```python
def predict_image(model, img_path):
    pass
```

---

## 📈 Evaluation
```python
model.evaluate(val_gen)
```

---

## 🔮 Future Improvements
- Transfer Learning
- Model tuning
- Deployment (Flask/FastAPI)
