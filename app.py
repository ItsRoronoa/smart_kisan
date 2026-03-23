from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import hashlib
import os
from io import BytesIO

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)

IMG_SIZE = 224
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image / 255.0  # Standard scaling
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or request.form
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"message": "Missing email or password"}), 400
        
    if os.path.exists("users.txt"):
        with open("users.txt", "r") as f:
            for line in f:
                stored_email, stored_password = line.strip().split(',')
                if email == stored_email and password == stored_password:
                    return jsonify({"message": "Login successful"})
                    
    return jsonify({"message": "Invalid email or password"}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or request.form
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"message": "Missing email or password"}), 400
        
    # Check if user already exists
    if os.path.exists("users.txt"):
        with open("users.txt", "r") as f:
            for line in f:
                if line.startswith(email + ","):
                    return jsonify({"message": "User already exists"}), 400
    
    # Save the new user to users.txt
    with open("users.txt", "a") as f:
        f.write(f"{email},{password}\n")
        
    return jsonify({"message": "Signup successful"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"result": "Error: No image found"})
        
    file = request.files['image']
    file_bytes = file.read()
    
    if not file_bytes:
        return jsonify({"result": "Error: Empty file"})

    # Check for duplicate data based on SHA-256 hash of the image file
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    save_path = os.path.join(UPLOAD_FOLDER, f"{file_hash}.jpg")
    
    if os.path.exists(save_path):
        return jsonify({
            "result": "Duplicate Data! This image has already been uploaded.", 
            "duplicate": True
            })
    
    # Save image as new data
    try:
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
    except Exception as e:
        return jsonify({"result": f"Error saving file: {str(e)}"})

    # Process image
    image = Image.open(BytesIO(file_bytes))
    processed = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Reject if the model doesn't have high enough confidence (e.g. not in model's training data)
    if confidence < 0.60:
        return jsonify({
            "result": "Error: Not found in model (Low confidence)",
            "class": -1,
            "confidence": confidence,
            "duplicate": False
        })

    # classes
    CLASS_NAMES = {
        0: "Pepper__bell___Bacterial_spot",
        1: "Pepper__bell___healthy",
        2: "Potato___Early_blight",
        3: "Potato___Late_blight",
        4: "Potato___healthy",
        5: "Tomato_Bacterial_spot",
        6: "Tomato_Early_blight",
        7: "Tomato_Late_blight",
        8: "Tomato_Leaf_Mold",
        9: "Tomato_Septoria_leaf_spot",
        10: "Tomato_Spider_mites_Two_spotted_spider_mite",
        11: "Tomato__Target_Spot",
        12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
        13: "Tomato__Tomato_mosaic_virus",
        14: "Tomato_healthy"
    }
    label_name = CLASS_NAMES.get(int(predicted_class), f"Class {int(predicted_class)}")

    return jsonify({
        "result": label_name,
        "class": int(predicted_class),
        "confidence": confidence,
        "duplicate": False
    })

if __name__ == '__main__':
    app.run(hpost="0.0.0.0", port=int(os.environ.get("PORT",10000)))