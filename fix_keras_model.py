import zipfile
import json
import os
import shutil

def remove_key(d, key_to_remove):
    if isinstance(d, dict):
        if key_to_remove in d:
            del d[key_to_remove]
        for k, v in d.items():
            remove_key(v, key_to_remove)
    elif isinstance(d, list):
        for item in d:
            remove_key(item, key_to_remove)

def fix_keras_file(filepath):
    # Rename original
    backup_path = filepath + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
    
    # Extract to temp
    temp_dir = "temp_keras_extract"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Read config.json
    config_path = os.path.join(temp_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Modify
    remove_key(config, 'quantization_config')
    
    # Write config.json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Zip back
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, temp_dir)
                zip_ref.write(abs_path, rel_path)
    
    shutil.rmtree(temp_dir)
    print("Successfully patched model.keras!")

if __name__ == "__main__":
    fix_keras_file("model.keras")

import tensorflow as tf
print("Testing load after patch...")
model = tf.keras.models.load_model("model.keras", compile=False)
print("Model loaded successfully!")
