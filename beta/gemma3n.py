# pip install accelerate

import os
import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText
import PIL
from PIL import Image # Import Image specifically

# Define the custom path to your .kaggle directory
# Replace 'C:/Your/Custom/Path/To/.kaggle' with the actual path
custom_kaggle_config_dir = ''
# Set the KAGGLE_CONFIG_DIR environment variable
os.environ['KAGGLE_CONFIG_DIR'] = custom_kaggle_config_dir

# Now, call the model_download function.
# kagglehub will look for kaggle.json inside the directory specified by KAGGLE_CONFIG_DIR
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

# print("Path to model files:", path)

# Optional: You might want to unset the environment variable if it's a temporary change
# del os.environ['KAGGLE_CONFIG_DIR']


processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

# Corrected line for loading the image
try:
    image_file = Image.open("img.png")
except FileNotFoundError:
    print("Error: 'img.png' not found. Please make sure the image file exists in the same directory as your script, or provide the full path to the image.")
    exit() # Exit the script if the image isn't found


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_file}, # Use the loaded image object here
            {"type": "text", "text": "Give text in key value pair"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(text[0])