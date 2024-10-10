from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow
import os


load_dotenv()

if not os.getenv("ROBOFLOW_API_KEY"):
    raise ValueError("Please set your ROBOFLOW_API_KEY in a .env file")

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

vehicle_detect_model = inference.get_model("cartypes-zywky/1")


def detect_vehicles(image):
    base64_image = image_to_base64(image)

    result = vehicle_detect_model.predict({"image": base64_image})

    return result