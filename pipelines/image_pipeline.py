import numpy as np
from PIL import Image
from roboflow.util.two_stage_utils import ocr_infer

from models.plate_detection import detect_license_plate, detect_plate
from models.vehicle_detection import detect_vehicles
from models.ocr import extract_license_plate_text
from preprocessing.image_preprocessing import preprocess_license_plate
from utils.custom_logs import setup_logger
from utils.file_utils import visualize_detections, visualize_combined_detections

logger = setup_logger("image_pipeline.py")

def detect_vehicle_plate(image):
    vehicle_detection_result = detect_vehicles(image)
    plate_detection_result = detect_plate(image)

    return vehicle_detection_result, plate_detection_result


