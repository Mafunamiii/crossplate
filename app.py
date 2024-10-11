import cv2
import streamlit as st
import numpy as np
from PIL import Image

from models.plate_detection import detect_plate
from models.vehicle_detection import detect_vehicles
from models.ocr import extract_license_plate_text
from pipelines.image_pipeline import detect_vehicle_plate
from preprocessing.image_preprocessing import crop_plate, preprocess_license_plate
from utils.custom_logs import setup_logger
from utils.file_utils import visualize_detections, visualize_combined_detections

logger = setup_logger("app.py")

logger.info("Starting the Vehicle Detection App")
st.title("CrossPlate - Vehicle Detection and License Plate Recognition App")
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

    vehicle_detected, plate_detected = detect_vehicle_plate(image)
    crop_image = crop_plate(image, plate_detected[0].predictions)
    pre_processed_image = preprocess_license_plate(crop_image)

    ocr_result = extract_license_plate_text(pre_processed_image)

    st.write("**License Plate Detected:** ", ocr_result)
    st.image(visualize_combined_detections(image, vehicle_detected, plate_detected), caption='Detection Summary', use_column_width=True)
    st.image(visualize_detections(image, vehicle_detected), caption='Vehicle Detection', use_column_width=True)
    st.image(visualize_detections(image, plate_detected), caption='Plate Detection', use_column_width=True)
    st.image(pre_processed_image, caption='Cropped Image', use_column_width=True)

