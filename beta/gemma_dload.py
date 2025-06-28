import os
import kagglehub

# Define the custom path to your .kaggle directory
# Replace 'C:/Your/Custom/Path/To/.kaggle' with the actual path
custom_kaggle_config_dir = ''

# Set the KAGGLE_CONFIG_DIR environment variable
os.environ['KAGGLE_CONFIG_DIR'] = custom_kaggle_config_dir

# Now, call the model_download function.
# kagglehub will look for kaggle.json inside the directory specified by KAGGLE_CONFIG_DIR
path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

print("Path to model files:", path)

# Optional: You might want to unset the environment variable if it's a temporary change
# del os.environ['KAGGLE_CONFIG_DIR']