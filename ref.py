# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 10:19:43 2025

@author: Babitha
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS
from pdf2image import convert_from_path
import tempfile
from paddleocr import PaddleOCR  # Import PaddleOCR

# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/2.jpg')

# Streamlit app title
st.title("Project: Fake Document Recognition")

# Load the model
model = load_model("Model.h5")

# Define image dimensions and categories
WIDTH, HEIGHT = 65, 65
categories = ['Original', 'Forgery']

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    image = np.array(image)
    
    # Ensure the image has 3 channels (RGB)
    if image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    test_image = cv2.resize(image, (WIDTH, HEIGHT))
    test_data = np.array(test_image, dtype="float") / 255.0
    test_data = test_data.reshape([-1, WIDTH, HEIGHT, 3])
    return image, test_data

# Function to segment the image using thresholding
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    segmented_image = cv2.bitwise_and(image, image, mask=thresholded)
    return segmented_image

# Function to extract text from an image using PaddleOCR
def extract_text_from_image_ocr(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR
    result = ocr.ocr(np.array(image), cls=True)  # OCR with layout analysis
    text = ""
    for line in result[0]:
        # Check if line[1] is a string before concatenating
        if isinstance(line[1], str):
            text += line[1] + "\n"
        else:
            # Handle cases where line[1] is not a string (optional)
            text += str(line[1]) + "\n"  # You can customize how to handle non-string values
    return text, result  # Return text and OCR result (bounding boxes)

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, ocr_result):
    # Convert the PIL image to a NumPy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Draw bounding boxes on the image
    for line in ocr_result[0]:
        points = line[0]  # The bounding box points
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        # Draw the bounding box (green color with thickness of 2)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

# Streamlit interface
st.write("Upload images or PDFs to get the predictions.")

# Upload file
uploaded_files = st.file_uploader("Choose images or PDFs...", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    predictions_list = []  # Initialize an empty list to store predictions
    
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':  # If it's a PDF
                # Create a temporary directory to save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())  # Save the uploaded file to the temp directory
                    tmp_file_path = tmp_file.name  # Get the path of the saved file
                
                # Convert PDF pages to images
                images = convert_from_path(tmp_file_path)
                
                # Process each page
                for i, image in enumerate(images):
                    # Extract text using PaddleOCR
                    text, ocr_result = extract_text_from_image_ocr(image)
                    
                    # Display the extracted text
                    st.write(f"Page {i + 1} Text Extracted:")
                    st.text(text)
                    
                    # Preprocess image and predict
                    test_image_o, test_data = load_and_preprocess_image(image)
                    
                    # Make prediction
                    pred = model.predict(test_data)
                    predictions = np.argmax(pred, axis=1)
                    
                    # Append prediction to the list
                    predictions_list.append(categories[predictions[0]])
                    
                    # Display the prediction and image
                    st.write(f'Prediction: {categories[predictions[0]]} for Page {i + 1}')
                    
                    # Show image with bounding boxes
                    image_with_boxes = draw_bounding_boxes(np.array(image), ocr_result)
                    st.image(image_with_boxes, caption=f"Image with OCR Bounding Boxes (Page {i + 1})", use_column_width=True)
                    
                    # Segment the image and show
                    segmented_image = segment_image(test_image_o)
                    fig = plt.figure()
                    fig.patch.set_facecolor('xkcd:white')
                    plt.title('Segmented Image')
                    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    st.pyplot(fig)
            
            else:  # If it's an image
                # Load image with PIL
                image = Image.open(uploaded_file)
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Extract text using PaddleOCR
                text, ocr_result = extract_text_from_image_ocr(image)
                
                # Display the extracted text
                st.write("Extracted Text from Image:")
                st.text(text)
                
                # Preprocess the image
                test_image_o, test_data = load_and_preprocess_image(image)
                
                # Make prediction
                pred = model.predict(test_data)
                predictions = np.argmax(pred, axis=1)
                
                # Append prediction to the list
                predictions_list.append(categories[predictions[0]])
                
                # Display the prediction
                st.write(f'Prediction: {categories[predictions[0]]}')
                
                # Display the image with bounding boxes
                image_with_boxes = draw_bounding_boxes(np.array(image), ocr_result)
                st.image(image_with_boxes, caption="Image with OCR Bounding Boxes", use_column_width=True)
                
                # Segment the image
                segmented_image = segment_image(test_image_o)
                
                # Display the segmented image
                fig = plt.figure()
                fig.patch.set_facecolor('xkcd:white')
                plt.title('Segmented Image')
                plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                st.pyplot(fig)
    
    # Display all predictions
    st.write("All Predictions:")
    st.write(predictions_list)
    
    # Convert predictions list to string and create audio
    predictions_text = ' '.join(predictions_list)
    language = 'en'
    speech = gTTS(text=predictions_text, lang=language, slow=False)
    speech.save("sample.mp3")
    audio_path = "sample.mp3"  # Replace with the path to your MP3 audio file

    st.audio(audio_path, format='audio/mp3')
else:
    st.write("Please upload image or PDF files.")
