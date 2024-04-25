from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load your trained model
loaded_model = tf.saved_model.load("C:/Users/UTKARSH/CSR-Detection_saved_model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = (file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    if file:
        try:
            # Read image file directly from the uploaded file object
            image = Image.open(file)
            # Preprocess image
            image = image.resize((150, 150))  # Resize according to your model's input
            image = np.array(image) / 255.0 
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
                image = np.repeat(image, 3, axis=-1) # Normalize pixel values
            image = np.expand_dims(image, axis=0)  # Model expects an array of images
            image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert dtype to float32
            # Make prediction
            prediction = loaded_model(image, training=False)
            # Example post-processing: Convert prediction to label (assuming binary classification)
            label = "Normal" if prediction[0][0] > 0.5 else "CSR"
            response = {
                'message': 'Prediction performed.',
                'result': label,
            }
            return jsonify(label)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Error processing file.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
