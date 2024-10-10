import cv2
import numpy as np
from PIL import Image

from utils.custom_logs import setup_logger

logger = setup_logger("image_preprocessing.py")

def crop_plate(image, detections):
    cropped_plates = []  # Initialize a list to hold cropped plates
    if isinstance(image, Image.Image):
        image = np.array(image)
    for detection in detections:  # Iterate over each detection
        x_center, y_center = detection.x, detection.y  # Access using attributes
        width, height = detection.width, detection.height  # Access using attributes

        # Calculate the top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Crop the region containing the license plate
        cropped_plate = image[y1:y2, x1:x2]
        cropped_plates.append(cropped_plate)  # Add the cropped plate to the list
    logger.info(f"Cropped plates: {cropped_plates}")

    return cropped_plate


def preprocess_license_plate(cropped_plate):
    logger.info(f"Preprocessing the license plate image... {cropped_plate}")

    # Convert the PIL image to a NumPy array
    cropped_plate = np.array(cropped_plate)
    logger.info(f"(preprocess_license_plate) Converted to NumPy array: {cropped_plate.shape}")

    # Check if the cropped plate is an RGBA image and convert to RGB
    if cropped_plate.shape[2] == 4:  # RGBA has 4 channels
        logger.info("(preprocess_license_plate) Converting RGBA to RGB.")
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_RGBA2RGB)

    # Check if the cropped plate is a valid image
    if cropped_plate is None or len(cropped_plate.shape) != 3 or cropped_plate.shape[2] != 3:
        logger.warning("Warning: Input image is not a valid color image.")
        return None

    # Resize image slightly for better processing
    resized_plate = cv2.resize(cropped_plate, (cropped_plate.shape[1] * 2, cropped_plate.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    logger.info(f"(preprocess_license_plate) Resized plate image: {resized_plate.shape}")

    # Convert to grayscale
    gray_plate = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2GRAY)
    logger.info(f"(preprocess_license_plate) Grayscale plate image: {gray_plate.shape}")

    # Apply histogram equalization to enhance contrast
    equalized_plate = cv2.equalizeHist(gray_plate)
    logger.info(f"(preprocess_license_plate) Equalized plate image: {equalized_plate.shape}")

    # Dilation to enhance the text slightly (you can experiment with iterations)
    kernel = np.ones((3, 3), np.uint8)
    dilated_plate = cv2.dilate(equalized_plate, kernel, iterations=0)
    logger.info(f"(preprocess_license_plate) Dilated plate image: {dilated_plate.shape}")

    return dilated_plate

