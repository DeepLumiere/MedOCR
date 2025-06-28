import os
import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

custom_kaggle_config_dir = ''
os.environ['KAGGLE_CONFIG_DIR'] = custom_kaggle_config_dir

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

try:
    image_file = Image.open("img.png")
except FileNotFoundError:
    print("Error: 'img.png' not found. Please make sure the image file exists in the same directory as your script, or provide the full path to the image.")
    exit()


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_file},
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