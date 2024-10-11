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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

        # Draw the bounding box in green (BGR format)
        logger.info(f"Drawing bounding box at ({x1}, {y1}) ({x2}, {y2})")

            # Optionally add the class label and confidence
        label = f"{detection.class_name} ({detection.confidence:.2f})"

        if detection.class_name == "licenseplate":
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif detection.class_name == "car":
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green in BGR
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    logger.info("Bounding boxes drawn successfully!")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB format
    return image  # Return the original image in BGR format


def visualize_combined_detections(image, vehicle_result, plate_result):
    logger.info("Visualizing combined detections for vehicles and license plates.")

    # Convert the image from PIL to OpenCV format if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convert PIL image to NumPy array
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw vehicle bounding boxes in green (BGR format)
    if isinstance(vehicle_result, list) and len(vehicle_result) > 0:
        vehicle_detections = vehicle_result[0]  # Assuming the first entry has the needed results
        for detection in vehicle_detections.predictions:
            # Get the bounding box coordinates
            x_center, y_center = detection.x, detection.y
            width, height = detection.width, detection.height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw bounding box in green for vehicles
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Car ({detection.confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw license plate bounding boxes in blue (BGR format)
    if isinstance(plate_result, list) and len(plate_result) > 0:
        plate_detections = plate_result[0]  # Assuming the first entry has the needed results
        for detection in plate_detections.predictions:
            # Get the bounding box coordinates
            x_center, y_center = detection.x, detection.y
            width, height = detection.width, detection.height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw bounding box in blue for license plates
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"License Plate ({detection.confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    logger.info("Combined bounding boxes drawn successfully!")

    # Convert back to RGB format for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB format
    return image  # Return the modified image with both bounding boxes

