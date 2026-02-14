import os
import cv2
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
from paddleocr import PaddleOCR
from gtts import gTTS
import tempfile
import base64
import streamlit as st
import random

# === Function to set background ===
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/jpeg;base64,{bin_str}");
    background-position: center;
    background-size: cover;
    }}
    </style>
    '''
    st.markdown('<style>h1 { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

# === Preprocess image to match model input (65x65) ===
def load_and_preprocess_image(image):
    image = np.array(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    resized = cv2.resize(image, (65, 65))  # Corrected to 65x65
    test_data = np.array(resized, dtype="float32") / 255.0
    test_data = test_data.reshape([-1, 65, 65, 3])
    return image, test_data

# === OCR using PaddleOCR ===
def extract_text_from_image_ocr(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(np.array(image), cls=True)
    text = ""
    for line in result[0]:
        if isinstance(line[1], str):
            text += line[1] + "\n"
        else:
            text += str(line[1]) + "\n"
    return text, result

# === Segment Image ===
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    segmented = cv2.bitwise_and(image, image, mask=thresh)
    return segmented

# === Convert PDF to Images ===
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# === Set up Streamlit App ===
set_background('background/2.jpg')
st.title("📄 Fake Document Recognition")

# Load the pre-trained model
try:
    model = load_model("Model.h5")
    # **IMPORTANT: SET THIS CORRECTLY BASED ON YOUR MODEL'S OUTPUT ORDER**
    # **UPDATE THIS LINE BASED ON YOUR MODEL TRAINING**
    categories = ['Original', 'Forgery']  # <--- ADJUST THIS LINE BASED ON YOUR TRAINING
except FileNotFoundError:
    st.error("Error: The model file 'Model.h5' was not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.write("Upload images or PDF files for fake document detection.")

# === Upload ===
uploaded_files = st.file_uploader("Choose image or PDF...", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    predictions_list = []

    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1].lower()

        if ext == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                images = pdf_to_images(tmp_path)
            except Exception as e:
                st.error(f"PDF conversion failed: {e}")
                continue

            for i, image in enumerate(images):
                st.subheader(f"📄 Page {i+1} - {uploaded_file.name}")
                try:
                    text, ocr_result = extract_text_from_image_ocr(image)
                    st.text_area("Extracted Text", text, height=150)

                    original_img, input_data = load_and_preprocess_image(image)
                    pred = model.predict(input_data)
                    predicted_index = np.argmax(pred)
                    result = categories[predicted_index]
                    predictions_list.append(result)

                    st.write(f"🧠 Prediction: {result}")
                    st.image(original_img, caption=f"Prediction: {result}", use_column_width=True)

                    segmented = segment_image(original_img)
                    st.image(segmented, caption="Segmented Image", use_column_width=True)

                    ocr_image_with_boxes = original_img.copy()
                    for box in ocr_result[0]:
                        points = np.array(box[0], dtype=np.int32)
                        cv2.polylines(ocr_image_with_boxes, [points], True, (0, 255, 0), 2)
                    st.image(ocr_image_with_boxes, caption="OCR with Bounding Boxes", use_column_width=True)

                except Exception as e:
                    st.error(f"Error processing page {i+1} of '{uploaded_file.name}': {e}")

        else:  # Handle image
            image = Image.open(uploaded_file)
            st.subheader(f"🖼️ {uploaded_file.name}")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            try:
                text, ocr_result = extract_text_from_image_ocr(image)
                st.text_area("Extracted Text", text, height=150)

                original_img, input_data = load_and_preprocess_image(image)
                pred = model.predict(input_data)
                predicted_index = np.argmax(pred)
                result = categories[predicted_index]
                predictions_list.append(result)

                st.write(f"🧠 Prediction: {result}")
                st.image(original_img, caption=f"Prediction: {result}", use_column_width=True)

                segmented = segment_image(original_img)
                st.image(segmented, caption="Segmented Image", use_column_width=True)

                ocr_image_with_boxes = original_img.copy()
                for box in ocr_result[0]:
                    points = np.array(box[0], dtype=np.int32)
                    cv2.polylines(ocr_image_with_boxes, [points], True, (0, 255, 0), 2)
                st.image(ocr_image_with_boxes, caption="OCR with Bounding Boxes", use_column_width=True)

            except Exception as e:
                st.error(f"Error processing image '{uploaded_file.name}': {e}")

    # === Text-to-Speech for Predictions ===
    predictions_text = ' '.join([f"{uploaded_files[i].name}: {predictions_list[i]}" for i in range(len(uploaded_files))])
    if predictions_text.strip():
        language = 'en'
        speech = gTTS(text=f"Predictions for uploaded documents: {predictions_text}", lang=language, slow=False)
        speech.save("sample.mp3")
        st.audio("sample.mp3", format='audio/mp3')
else:
    st.write("Please upload image or PDF files.")