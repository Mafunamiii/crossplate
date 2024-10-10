import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.custom_logs import setup_logger

logger = setup_logger("file_utils.py")


def image_to_base64(image):
    logger.info(f"image: {image}")


    if image is None:
        raise ValueError("No image provided")

    # Determine the image format
    image_format = image.format if image.format else 'PNG'  # Default to PNG if format is None

    buffered = BytesIO()
    try:
        image.save(buffered, format=image_format)
        buffered.seek(0)

        # Convert to base64
        img_str = base64.b64encode(buffered.read()).decode('utf-8')
        return img_str
    except Exception as e:
        print(f"Error saving image: {e}")
        raise


def visualize_detections(image, results):
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
    else:
        logger.error("No detection results found.")
        return image

    logger.info(f"Visualizing the detections...\n{result}")

    # Convert the image from PIL to OpenCV format if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convert PIL image to NumPy array

    # Loop through the predictions
    for detection in result.predictions:  # Access predictions from the response
        # Get the bounding box coordinates
        x_center, y_center = detection.x, detection.y
        width, height = detection.width, detection.height

        # Calculate the top-left corner of the bounding box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)

        # Calculate the bottom-right corner of the bounding box
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box
        logger.info(f"Drawing bounding box at ({x1}, {y1}) ({x2}, {y2})")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optionally add the class label and confidence
        label = f"{detection.class_name} ({detection.confidence:.2f})"
        logger.info(f"Adding label: {label}")
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    logger.info("Bounding boxes drawn successfully!")

    # Convert the image from BGR (OpenCV default) to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info("Conversion to RGB successful!")

    return image_rgb

def visualize_combined_detections(original_image, car_result, plate_result):
    # Load the image
    logger.info("Visualizing the combined detections...")

    # Function to draw bounding boxes
    def draw_bounding_box(original_image, detection, color, label):
        logger.info(f"Drawing bounding box for {label} detection...")
        # Get the bounding box coordinates
        x_center, y_center = detection.x, detection.y
        width, height = detection.width, detection.height
        logger.info(f"Bounding box coordinates: ({x_center}, {y_center}) ({width}, {height})")
        # Calculate the top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        logger.info(f"Bounding box corners: ({x1}, {y1}) ({x2}, {y2})")
        # Draw the bounding box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
        logger.info(f"Bounding box drawn successfully!")
        # Add the label and confidence score
        confidence = detection.confidence
        label_with_conf = f"{label} ({confidence:.2f})"
        cv2.putText(original_image, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    logger.info(f"Drawing bounding boxes for car and license plate detections...")
    logger.info(f"Car detection results: {car_result}")
    logger.info(f"License plate detection results: {plate_result}")

    # Draw car detection (in green)
    for car_detection in car_result:
        for detection in car_detection.predictions:
            logger.info("Drawing bounding box for car detection...")
            draw_bounding_box(original_image, detection, (0, 255, 0), "Car")

    # Draw license plate detection (in yellow)
    for plate_detection in plate_result:
        for detection in plate_detection.predictions:
            logger.info("Drawing bounding box for license plate detection...")
            draw_bounding_box(original_image, detection, (255, 255, 0), "License Plate")

    # Convert the image from BGR (OpenCV default) to RGB for displaying with matplotlib
    logger.info("Conversion to RGB...")
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    logger.info("Conversion successful!")
    return image_rgb
