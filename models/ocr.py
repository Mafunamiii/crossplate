import pytesseract

from utils.custom_logs import setup_logger

logger = setup_logger("ocr.py")

def extract_license_plate_text(cropped_plate):
    logger.info(f"(extract_license_plate_text)Extracting text from license plate... {cropped_plate}")
    plate_text = pytesseract.image_to_string(cropped_plate, config='--psm 7')
    logger.info(f"(extract_license_plate_text)Text extracted successfully! {plate_text}")
    return plate_text.strip()