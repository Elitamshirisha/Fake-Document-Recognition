# Fake Document Recognition System

## Project Overview

Fake Document Recognition is a deep learning–based system designed to detect whether a document is **Original** or **Forgery**.
The application uses a trained convolutional neural network model to analyze document images and PDFs. It also includes OCR-based text extraction and a voice summary of the prediction.

The system is developed to help organizations, institutions, and recruiters verify the authenticity of documents quickly and accurately.

## Features

### Document Classification

* Detects whether a document is **Original** or **Forgery**.
* Uses a deep learning model (DenseNet-based architecture).

### Multi-format Support

* Accepts both **image files** (JPG, PNG, JPEG).
* Accepts **PDF documents** and converts them to images for analysis.

### OCR Text Extraction

* Extracts text from uploaded documents using PaddleOCR.
* Displays detected text with bounding boxes.

### Image Segmentation

* Highlights important regions of the document.
* Helps visualize features used for prediction.

### Voice-based Prediction Summary

* Converts the prediction result into speech using text-to-speech.

---

## Technologies Used

### Frontend

* Streamlit (Web interface)

### Backend / Model

* Python
* TensorFlow / Keras
* DenseNet121 (Deep Learning Model)

### Image & OCR Processing

* OpenCV
* PaddleOCR
* PyMuPDF (PDF to image conversion)



## How to Run

### Step 1: Clone the repository

git clone https://github.com/Elitamshirisha/Fake-Document-Recognition.git


### Step 2: Install dependencies

pip install -r requirements.txt


### Step 3: Run the application

streamlit run app.py




## Output

The system predicts:

Original Document

Forgery Document

