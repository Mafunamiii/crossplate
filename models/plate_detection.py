from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow
import os
from inference import get_model

from utils.custom_logs import setup_logger
from utils.file_utils import image_to_base64

logger = setup_logger("plate_detection.py")

load_dotenv()

if not os.getenv("ROBOFLOW_API_KEY"):
    raise ValueError("Please set your ROBOFLOW_API_KEY in a .env file")

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

license_detect_model = get_model("yolov7-license-plate-detection/3")

def detect_plate(image):
    logger.info("Converting image to base64...")
    base64_image = image_to_base64(image)
    logger.info("Image converted to base64 successfully!")
    result = license_detect_model.infer(base64_image)
    logger.info("Detection completed successfully!")
    return result

def detect_license_plate(uncropped_vehicle_image,cropped_vehicle_image):
    logger.info(f"(detect_license_plate) Converting images to base64...{uncropped_vehicle_image} and {cropped_vehicle_image}")
    base64_image_uncropped = image_to_base64(uncropped_vehicle_image)
    base64_image_cropped = image_to_base64(cropped_vehicle_image)
    logger.info("(detect_license_plate Image converted to base64 successfully!")
    result_uncropped = license_detect_model.infer(base64_image_uncropped)
    result_cropped = license_detect_model.infer(base64_image_cropped)
    logger.info(f"(detect_license_plate) Detection completed successfully! {result_uncropped} and {result_cropped}")
    return result_uncropped, result_cropped
