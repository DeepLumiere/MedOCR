import os
import json
import re
import pytesseract
import cv2
import kagglehub # Keep this if you plan to download the original model

# --- Configuration for Model Caching ---
# Define your desired models directory (relative to the script, or absolute)
MODELS_DIR = "./models"
# Create the models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"DEBUG: Ensuring models directory exists at: {os.path.abspath(MODELS_DIR)}")

# Set environment variables for Hugging Face and KaggleHub to use the custom directory
# These *must* be done BEFORE importing AutoProcessor or AutoModelForImageTextToText,
# and before calling kagglehub.model_download.

# For Hugging Face Transformers (will affect AutoProcessor and AutoModelForImageTextToText)
os.environ['HF_HOME'] = MODELS_DIR
# For Hugging Face Hub (where actual files are cached within HF_HOME)
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(MODELS_DIR, 'hub')

# For KaggleHub (specifically the model download cache location)
os.environ['KAGGLEHUB_CACHE'] = os.path.join(MODELS_DIR, 'kagglehub')
print(f"DEBUG: HF_HOME set to: {os.environ.get('HF_HOME')}")
print(f"DEBUG: HUGGINGFACE_HUB_CACHE set to: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
print(f"DEBUG: KAGGLEHUB_CACHE set to: {os.environ.get('KAGGLEHUB_CACHE')}")

# --- Configuration for Kaggle API Credentials (kaggle.json) ---
# If kaggle.json is in the same directory as this script:
# Use os.path.dirname(os.path.abspath(__file__)) to get the script's directory.
custom_kaggle_config_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['KAGGLE_CONFIG_DIR'] = custom_kaggle_config_dir
print(f"DEBUG: KAGGLE_CONFIG_DIR set to: {os.environ.get('KAGGLE_CONFIG_DIR')}")


# Make sure this line is correct for your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"R:\DeepWorks\DeepPython\MedOCR\tesseract\tesseract.exe"


def preprocess_image_for_ocr(image_path):
    # Your existing code for preprocessing
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


def correct_common_ocr_errors(text):
    corrected_text = text
    corrected_text = re.sub(r'(\d)(\d{2})\s*(U/L|mmol/L|mg/dL|g/dL)', r'\1.\2 \3', corrected_text)
    corrected_text = re.sub(r'MCHC:\s*(\d{2})0', r'MCHC: \1.0', corrected_text)
    print(f"DEBUG: Applied rule-based OCR corrections. Original length: {len(text)}, Corrected length: {len(corrected_text)}")
    return corrected_text


# --- Gemma Model Loading ---
from transformers import AutoProcessor, AutoModelForImageTextToText # Ensure this import is here

processor, model = None, None
FINE_TUNED_MODEL_PATH = "./fine_tuned_gemma_medical_extractor" # Path where you save your fine-tuned model


try:
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        print(f"DEBUG: Loading fine-tuned Gemma model from {FINE_TUNED_MODEL_PATH}")
        processor = AutoProcessor.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForImageTextToText.from_pretrained(FINE_TUNED_MODEL_PATH, torch_dtype="auto", device_map="auto")
        print("DEBUG: Fine-tuned Gemma model loaded successfully.")
    else:
        # Fallback to original KaggleHub download if fine-tuned model not found
        print("DEBUG: Fine-tuned model not found. Attempting to load original Gemma 3n from KaggleHub.")
        # KaggleHub's cache location is now controlled by KAGGLEHUB_CACHE environment variable set above
        # Kaggle's API authentication (kaggle.json) location is now controlled by KAGGLE_CONFIG_DIR
        GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

        processor = AutoProcessor.from_pretrained(GEMMA_PATH)
        model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")
        print(f"DEBUG: Gemma 3n model loaded successfully from {GEMMA_PATH}.")

except Exception as e:
    print(f"ERROR: Could not load Gemma model. Error: {e}")
    processor, model = None, None


def extract_json_with_gemma(text_to_analyze,
                            query_text="Given the following medical lab report text, extract all relevant information into a JSON object. Use standardized medical key names like 'PatientName', 'BP', 'HeartRate', 'Glucose', 'MCHC', etc. Ensure numerical values are parsed as numbers (float or int). If a value is not present, use null. Example: {'PatientName': 'John Doe', 'BP': '120/80', 'HeartRate': 75, 'Glucose': 90.5}"):
    if processor is None or model is None:
        print("Gemma model not loaded. Skipping Gemma extraction.")
        return None

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
                                 disable_compile=True)
        gemma_response_text = processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        json_match = re.search(r'```json\n(.*?)\n```', gemma_response_text[0], re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = gemma_response_text[0].strip()

        try:
            extracted_json = json.loads(json_string)
            print("DEBUG: Successfully parsed Gemma's output as JSON.")
            return extracted_json
        except json.JSONDecodeError as e:
            print(f"DEBUG: Gemma output is not valid JSON. Attempting direct parsing or returning raw text. Error: {e}")
            print(f"DEBUG: Gemma raw output: {json_string}")
            return {"gemma_raw_output": json_string}

    except Exception as e:
        print(f"ERROR: During Gemma text generation or parsing: {e}")
        return None


# --- Key standardization functions ---
KEY_MAPPINGS = {
    "gluc": "Glucose",
    "glucose": "Glucose",
    "bp": "BloodPressure",
    "heartrate": "HeartRate",
    "mchc": "MCHC",
    "patientname": "PatientName",
    # Add more mappings as needed
}

def standardize_keys(data, key_mappings):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            standardized_key = key_mappings.get(key.lower(), key)
            new_data[standardized_key] = standardize_keys(value, key_mappings)
        return new_data
    elif isinstance(data, list):
        return [standardize_keys(item, key_mappings) for item in data]
    else:
        return data


def process_lab_report_single_image(image_path):
    print(f"\n--- Starting processing for image: {image_path} ---")

    raw_text = extract_text_from_image_with_pytesseract(image_path)

    if not raw_text:
        print("PyTesseract failed to extract any text. Cannot proceed with Gemma extraction.")
        return json.dumps({"status": "pytesseract_text_extraction_failed"}, indent=4)

    corrected_raw_text = correct_common_ocr_errors(raw_text)

    print("\n--- Raw Text Extracted by PyTesseract (with initial corrections) ---")
    print(corrected_raw_text)
    print("------------------------------------------")

    print("\n--- Sending text to Gemma 3n for JSON extraction ---")
    gemma_parsed_data = extract_json_with_gemma(corrected_raw_text)

    if gemma_parsed_data:
        standardized_gemma_data = standardize_keys(gemma_parsed_data, KEY_MAPPINGS)

        final_output = {
            "image_processed": image_path,
            "pytesseract_raw_text": raw_text,
            "corrected_pytesseract_text": corrected_raw_text,
            "gemma_extracted_data": standardized_gemma_data
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
            "pytesseract_raw_text": raw_text,
            "corrected_pytesseract_text": corrected_raw_text
        }, indent=4)


# Main execution block
image_file_to_process = 'img.png'
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