import os
import io
import json
import re
from PIL import Image # Used for opening images with pytesseract
import pytesseract    # Python wrapper for Tesseract OCR
import cv2            # OpenCV for image preprocessing (can leverage GPU if compiled with CUDA)
import numpy as np    # For converting PIL images to OpenCV format


pytesseract.pytesseract.tesseract_cmd = r"R:\DeepWorks\DeepPython\MediOCRe\tesseract\tesseract.exe"

def preprocess_image_for_ocr(image_path):

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"DEBUG: Error: Could not load image from {image_path}. Check if file exists and is a valid image format.")
            return None

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        preprocessed_img = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )


        print(f"DEBUG: Image preprocessed for OCR: {image_path}")
        return preprocessed_img

    except Exception as e:
        print(f"DEBUG: Error during image preprocessing for {image_path}: {e}")
        return None


def extract_text_from_image(image_path):

    try:
        preprocessed_img = preprocess_image_for_ocr(image_path)
        if preprocessed_img is None:
            print(f"DEBUG: Preprocessing failed for {image_path}. Cannot proceed with OCR.")
            return None

        print(f"DEBUG: Attempting OCR on preprocessed image from {image_path}...")
        raw_text = pytesseract.image_to_string(preprocessed_img, lang='eng', config='--psm 3')

        if raw_text.strip():
            print(f"DEBUG: Successfully extracted text from: {image_path}")
            return raw_text
        else:
            print(f"DEBUG: No text detected by Tesseract in: {image_path}. This could be due to poor image quality or incorrect Tesseract configuration.")
            return None

    except pytesseract.TesseractNotFoundError:
        print(f"ERROR: Tesseract is not installed or not in your system's PATH.")
        print(f"Please ensure Tesseract is installed and its executable path is correctly set,")
        print(f"either in system PATH or by uncommenting/setting 'pytesseract.pytesseract.tesseract_cmd'.")
        return None
    except Exception as e:
        print(f"DEBUG: Error extracting text from {image_path}: {e}")
        return None

def parse_medical_data(raw_text):

    if not raw_text:
        print("DEBUG: Raw text is empty, cannot parse medical data.")
        return {}

    extracted_data = {}

    bp_pattern = re.compile(r'(?:BP|Blood Pressure|BloodPressure)[:\s]*(\d{2,3}/\d{2,3})\b', re.IGNORECASE)
    bp_match = bp_pattern.search(raw_text)
    if bp_match:
        extracted_data['BP'] = bp_match.group(1)
        print(f"DEBUG: Found BP: {extracted_data['BP']}")
    else:
        print("DEBUG: BP pattern not found.")


    hr_pattern = re.compile(r'(?:HR|Heart Rate|HeartRate)[:\s]*(\d{2,3})\s*(?:bpm|beats per minute)?\b', re.IGNORECASE)
    hr_match = hr_pattern.search(raw_text)
    if hr_match:
        try:
            extracted_data['HeartRate'] = int(hr_match.group(1)) # Convert to int if numerical
            print(f"DEBUG: Found HeartRate: {extracted_data['HeartRate']}")
        except ValueError:
            print(f"DEBUG: Could not convert HeartRate '{hr_match.group(1)}' to integer.")
    else:
        print("DEBUG: Heart Rate pattern not found.")


    glucose_pattern = re.compile(r'(?:Glucose|Blood Sugar|BloodSugar)[:\s]*(\d+\.?\d*)\s*(?:mg/dL|mmol/L)?\b', re.IGNORECASE)
    glucose_match = glucose_pattern.search(raw_text)
    if glucose_match:
        try:
            extracted_data['Glucose'] = float(glucose_match.group(1)) # Convert to float
            print(f"DEBUG: Found Glucose: {extracted_data['Glucose']}")
        except ValueError:
            print(f"DEBUG: Could not convert Glucose '{glucose_match.group(1)}' to float.")
    else:
        print("DEBUG: Glucose pattern not found.")


    name_pattern = re.compile(r'(?:Patient Name|Name)[:\s]*([A-Za-z\s\.]+\b)', re.IGNORECASE)
    name_match = name_pattern.search(raw_text)
    if name_match:
        extracted_data['PatientName'] = name_match.group(1).strip()
        print(f"DEBUG: Found PatientName: {extracted_data['PatientName']}")
    else:
        print("DEBUG: Patient Name pattern not found.")

    return extracted_data

def process_lab_report(image_path):

    print(f"\nProcessing lab report: {image_path}")
    raw_text = extract_text_from_image(image_path)

    if raw_text:
        print("\n--- Raw Extracted Text ---")
        print(raw_text)
        print("--------------------------")

        medical_data = parse_medical_data(raw_text)
        if medical_data:
            json_output = json.dumps(medical_data, indent=4)
            print("\n--- Extracted Data (JSON) ---")
            print(json_output)
            return json_output
        else:
            print("No specific medical data parsed from the text.")
            return json.dumps({"status": "no_medical_data_parsed", "raw_text": raw_text}, indent=4)
    else:
        print("Could not extract text. Skipping data parsing.")
        return json.dumps({"status": "text_extraction_failed"}, indent=4)


#Running the Tesseract
sample_image_path = 'test.png'

print(f"DEBUG: Checking for image file at: {os.path.abspath(sample_image_path)}")
if not os.path.exists(sample_image_path):
    print("Error, couldn't run the program")
else:
    print(f"DEBUG: Image file '{sample_image_path}' found.")
    result_json = process_lab_report(sample_image_path)
    if result_json:
        print(f"\nFinal result for {sample_image_path}:\n{result_json}")
    else:
        print(f"Failed to process {sample_image_path}.")
