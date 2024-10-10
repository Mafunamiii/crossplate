import cv2
import streamlit as st
import numpy as np
from PIL import Image

from models.vehicle_detection import detect_vehicles
from utils.custom_logs import setup_logger
from utils.file_utils import visualize_detections

logger = setup_logger("app.py")

logger.info("Starting the Vehicle Detection App")
st.title("Vehicle Detection App")
st.write("Upload an image to detect vehicles using YOLOv5 model.")

logger.info("Waiting for image upload...")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    logger.info("Image uploaded successfully!")
    image = Image.open(uploaded_file)
    logger.info("Image opened successfully!")

    st.write("Processing...")

    # Call the detection model
    logger.info("Calling the detection model...")
    detection_result = detect_vehicles(image)
    logger.info(f"Detection result: {detection_result}")

    if not detection_result:
        logger.error("No detection results to visualize.")
        st.error("No detections found.")
    else:
        logger.info("Visualizing the detection result...")
        image_result = visualize_detections(image, detection_result)

        # Display both images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.image(image_result, caption='Detection Result', use_column_width=True)

        logger.info("Detection displayed successfully!")

