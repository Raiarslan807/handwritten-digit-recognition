
# Handwritten Digit Recognition Using CNN

An intelligent handwritten digit recognition system developed using a Custom Convolutional Neural Network (CNN), TensorFlow, Keras, and Streamlit.

The application allows users to draw handwritten digits or upload digit images and receive real-time predictions with confidence scores.

---

## University Information

**University:** University of the Punjab  
**Campus:** Pothohar Campus, Gujar Khan  
**Department:** Information Technology  
**Program:** BS Information Technology  
**Semester:** 6th Semester  

## Project Information

**Project Title:** Handwritten Digit Recognition Using CNN  
**Submitted By:** Arsalan Ramzan  
**Algorithm:** Custom Convolutional Neural Network (CNN)  
**Dataset:** MNIST  
**Framework:** TensorFlow / Keras  

---

## 1. Project Overview

Handwritten Digit Recognition is a machine learning project that identifies handwritten numerical digits from images.

The system uses a Convolutional Neural Network (CNN) trained on the MNIST dataset. It recognizes digits from 0 to 9 and provides prediction confidence and class probability distributions.

A Streamlit-based graphical user interface allows users to interact with the trained model by drawing digits or uploading images.

## 2. Project Objectives

- Develop a deep learning model for handwritten digit classification.
- Understand and implement CNN architecture.
- Perform image preprocessing and normalization.
- Train and evaluate a custom CNN using the MNIST dataset.
- Build an interactive application for real-time digit recognition.
- Display prediction results and confidence scores.

## 3. Dataset Description

The project uses the MNIST handwritten digits dataset.

| Property | Description |
|---|---|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Dimensions | 28 × 28 pixels |
| Color Mode | Grayscale |
| Number of Classes | 10 |
| Classes | 0–9 |
| Pixel Range | 0–255 |

Each image represents a handwritten numerical digit.

## 4. Data Preprocessing

The following preprocessing operations were performed:

1. Pixel normalization from 0–255 to 0–1.
2. Reshaping images into CNN-compatible dimensions.
3. Splitting training data into training and validation sets.
4. Applying data augmentation using rotation, translation, and zoom.
5. Preparing images in grayscale format.

For user-provided images, additional preprocessing is performed to crop the digit, resize it, center it on a 28 × 28 canvas, and normalize pixel values.

## 5. CNN Model Architecture

A custom Convolutional Neural Network was developed for digit classification.

The architecture contains:

- Input Layer: 28 × 28 × 1
- Convolutional Layer: 32 filters
- Batch Normalization
- ReLU Activation
- Max Pooling
- Convolutional Layer: 64 filters
- Batch Normalization
- ReLU Activation
- Max Pooling
- Convolutional Layer: 128 filters
- Batch Normalization
- ReLU Activation
- Global Average Pooling
- Dense Layer: 128 neurons
- Dropout: 0.4
- Output Layer: 10 neurons with Softmax activation

**Total Parameters:** 111,146  
**Trainable Parameters:** 110,698  
**Non-trainable Parameters:** 448  

## 6. Model Training

The model was trained using the following configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 128 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 5 |
| Best Validation Accuracy | 99.167% |

Callbacks including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau were used during training.

## 7. Model Evaluation

The trained CNN was evaluated on the separate MNIST test dataset.

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 99.08% |
| Test Loss | 0.0273 |
| Correct Predictions | 9,908 |
| Incorrect Predictions | 92 |
| Macro F1-Score | 0.9908 |
| Weighted F1-Score | 0.9908 |

These results represent performance on the MNIST test dataset. Performance on external handwritten images may vary.

## 8. Application Features

The Streamlit application provides:

### Draw a Digit
Users can draw a digit between 0 and 9 using an interactive drawing canvas.

### Upload an Image
Users can upload handwritten digit images in PNG, JPG, or JPEG format.

### Prediction Results
The application displays:

- Predicted digit
- Prediction confidence
- Processed model input
- Class probability distribution
- MNIST test accuracy

### Interactive Interface
The application includes a customized university-themed interface and project information.

## 9. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- Streamlit Drawable Canvas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 10. Project Structure

```text
Handwritten-Digit-Recognition/
│
├── assets/
│   └── pu_logo.png
│
├── dataset/
│
├── models/
│   └── final_digit_recognition.keras
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── prediction.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### Step 2: Navigate to Project Directory

```bash
cd Handwritten-Digit-Recognition
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 12. Limitations

- The model is trained primarily on the MNIST dataset.
- It is designed to recognize one digit at a time.
- Predictions on unusual handwriting, noisy images, or different writing styles may vary.
- Images should have sufficient contrast and a clearly visible digit.

## 13. Conclusion

This project demonstrates the practical application of deep learning and convolutional neural networks for handwritten digit recognition.

The custom CNN achieved 99.08% accuracy on the MNIST test dataset. An interactive Streamlit application was also developed to demonstrate real-time predictions through drawing and image uploading.

The project provides practical experience in dataset preparation, image preprocessing, CNN architecture, model training, evaluation, and application development.

---

**Developed By:** Arsalan Ramzan  
**University:** University of the Punjab, Pothohar Campus, Gujar Khan  
**Program:** BS Information Technology — 6th Semester
