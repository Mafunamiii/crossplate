import streamlit as st
import numpy as np
from PIL import Image

from models.vehicle_detection import detect_vehicles
from utils.custom_logs import setup_logger

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

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Processing...")

    # Call the detection model\
    logger.info("Calling the detection model...")
    detection_result = detect_vehicles(image)

    # Display the bounding boxes or results (for now, we'll print raw detection result)

    logger.info("Displaying the detection:")
    st.write(detection_result)
    logger.info("Detection displayed successfully!")
