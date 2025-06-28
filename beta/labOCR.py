import os
import io
import json
import re
from PIL import Image  # Used for opening images with pytesseract
import pytesseract  # Python wrapper for Tesseract OCR
import cv2  # OpenCV for image preprocessing (can leverage GPU if compiled with CUDA)
import numpy as np  # For converting PIL images to OpenCV format
import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText

# --- PyTesseract Setup ---
pytesseract.pytesseract.tesseract_cmd = r"R:\DeepWorks\DeepPython\MedOCR\tesseract\tesseract.exe"


def preprocess_image_for_ocr(image_path):
    """
    Preprocesses an image for OCR by converting to grayscale and applying adaptive thresholding.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(
                f"DEBUG: Error: Could not load image from {image_path}. Check if file exists and is a valid image format.")
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


def extract_text_from_image_with_pytesseract(image_path):
    """
    Extracts raw text from an image using PyTesseract after preprocessing.
    """
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
            print(
                f"DEBUG: No text detected by Tesseract in: {image_path}. This could be due to poor image quality or incorrect Tesseract configuration.")
            return None
    except pytesseract.TesseractNotFoundError:
        print(f"ERROR: Tesseract is not installed or not in your system's PATH.")
        print(f"Please ensure Tesseract is installed and its executable path is correctly set,")
        print(f"either in system PATH or by uncommenting/setting 'pytesseract.pytesseract.tesseract_cmd'.")
        return None
    except Exception as e:
        print(f"DEBUG: Error extracting text from {image_path}: {e}")
        return None


# --- Gemma 3n Setup ---
# Define the custom path to your .kaggle directory
# Replace '' with the actual path if you've moved your kaggle.json
custom_kaggle_config_dir = ''
os.environ['KAGGLE_CONFIG_DIR'] = custom_kaggle_config_dir

processor, model = None, None  # Initialize to None
try:
    GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
    processor = AutoProcessor.from_pretrained(GEMMA_PATH)
    model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")
    print("DEBUG: Gemma 3n model loaded successfully.")
except Exception as e:
    print(
        f"ERROR: Could not load Gemma 3n model. Make sure you have downloaded it via KaggleHub and your kaggle.json is correctly configured. Error: {e}")


def extract_json_with_gemma(text_to_analyze,
                            query_text="Given the following text, extract all relevant medical information into a JSON object with appropriate keys and values. Ensure numbers are parsed as numerical types. If a value is missing, use null. Example: {'PatientName': 'John Doe', 'BP': '120/80', 'HeartRate': 75, 'Glucose': 90.5}"):
    """
    Feeds extracted text into Gemma 3n to get structured JSON data.
    """
    if processor is None or model is None:
        print("Gemma model not loaded. Skipping Gemma extraction.")
        return None

    # Gemma expects messages in a specific format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_text + "\n\nText: " + text_to_analyze}
            ]
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(**inputs, max_new_tokens=1024,
                                 disable_compile=True)  # Increased max_new_tokens for potentially larger JSON output
        gemma_response_text = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Gemma's response might include conversational filler, try to extract the JSON part
        json_match = re.search(r'```json\n(.*?)\n```', gemma_response_text[0], re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            # If no code block, try to parse the whole response directly if it looks like JSON
            json_string = gemma_response_text[0].strip()

        try:
            extracted_json = json.loads(json_string)
            print("DEBUG: Successfully parsed Gemma's output as JSON.")
            return extracted_json
        except json.JSONDecodeError as e:
            print(f"DEBUG: Gemma output is not valid JSON. Attempting direct parsing or returning raw text. Error: {e}")
            print(f"DEBUG: Gemma raw output: {json_string}")
            # If it's not strictly JSON, we can return the raw text for manual inspection
            return {"gemma_raw_output": json_string}

    except Exception as e:
        print(f"ERROR: During Gemma text generation or parsing: {e}")
        return None


def process_lab_report_single_image(image_path):
    """
    Main function to process a single lab report image:
    1. Extracts raw text using PyTesseract.
    2. Feeds the raw text to Gemma 3n for structured JSON extraction.
    """
    print(f"\n--- Starting processing for image: {image_path} ---")

    # Step 1: Extract raw text using PyTesseract
    raw_text = extract_text_from_image_with_pytesseract(image_path)

    if not raw_text:
        print("PyTesseract failed to extract any text. Cannot proceed with Gemma extraction.")
        return json.dumps({"status": "pytesseract_text_extraction_failed"}, indent=4)

    print("\n--- Raw Text Extracted by PyTesseract ---")
    print(raw_text)
    print("------------------------------------------")

    # Step 2: Feed raw text to Gemma 3n for JSON extraction
    print("\n--- Sending text to Gemma 3n for JSON extraction ---")
    gemma_parsed_data = extract_json_with_gemma(raw_text)

    if gemma_parsed_data:
        final_output = {
            "image_processed": image_path,
            "pytesseract_raw_text": raw_text,
            "gemma_extracted_data": gemma_parsed_data
        }
        print("\n--- Final Combined Extracted Data (JSON) ---")
        json_output = json.dumps(final_output, indent=4)
        print(json_output)
        print("--------------------------------------------")
        return json_output
    else:
        print("Gemma 3n failed to extract structured data from the text.")
        return json.dumps({
            "status": "gemma_extraction_failed",
            "image_processed": image_path,
            "pytesseract_raw_text": raw_text
        }, indent=4)


# --- Running the Combined Process ---
image_file_to_process = 'img.png'  # Use your single image here

print(f"DEBUG: Checking for image file at: {os.path.abspath(image_file_to_process)}")
if not os.path.exists(image_file_to_process):
    print(f"Error: Image file '{image_file_to_process}' not found. Please ensure it's in the same directory.")
else:
    print(f"DEBUG: Image file '{image_file_to_process}' found.")
    result_json = process_lab_report_single_image(image_file_to_process)
    if result_json:
        print(f"\nProcess completed for {image_file_to_process}.")
    else:
        print(f"Failed to fully process {image_file_to_process}.")