from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("classifier_model.h5")
print("Expected input shape:", model.input_shape)  # Debugging input shape

# Define a dictionary to map indices to disease names
disease_classes = {
    0: 'Apple_ceder',
    1: 'Apple_rust',
    2: 'Apple_multiple disease',
    3: 'Apple_Scab',
    4: 'Apple_healthy',
    5: 'Grape_cedar',
    6: 'Grape_rust',
    7: 'Grape_multiple disease',
    8: 'Grape_scarb',
    9: 'Grape_healthy'
    
}

# Preprocess the image according to model input requirements
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match the required dimensions
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = img.flatten()  # Flatten image to match model input shape (None, 8)
    return img[:8].reshape(1, -1)  # Select the first 8 features and reshape

print("Processed image shape example:", preprocess_image(Image.new('RGB', (224, 224))).shape)

# Initialize count variable

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            disease_name = disease_classes.get(predicted_class, "Unknown")

            return jsonify({
                "prediction_index": int(predicted_class),
                "disease_name": disease_name
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file"}), 400
if __name__ == '__main__':
    app.run(debug=True)
