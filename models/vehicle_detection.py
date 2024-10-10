from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow
import os
from inference import get_model

from utils.custom_logs import setup_logger
from utils.file_utils import image_to_base64

logger = setup_logger("vehicle_detection.py")

load_dotenv()

if not os.getenv("ROBOFLOW_API_KEY"):
    raise ValueError("Please set your ROBOFLOW_API_KEY in a .env file")

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

vehicle_detect_model = get_model("cartypes-zywky/1")


def detect_vehicles(image):
    logger.info("Converting image to base64...")
    base64_image = image_to_base64(image)
    logger.info("Image converted to base64 successfully!")
    result = vehicle_detect_model.infer(base64_image)
    logger.info("Detection completed successfully!")
    return result