import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils.custom_logs import setup_logger

logger = setup_logger("file_utils.py")

def image_to_base64(image):
    logger.info("Converting image to base64...")

    image_format = image.format
    logger.info(f"Detected image format: {image_format}")

    buffered = BytesIO()
    logger.info("Creating a BytesIO object...")
    image.save(buffered, format=image_format)
    logger.info("Image saved to BytesIO object successfully!")
    image_bytes = buffered.getvalue()
    logger.info("Image converted to bytes successfully!")
    return base64.b64encode(image_bytes).decode('utf-8')


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