from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('brain_tumor_model.h5')

@app.route('/')
def home():
    return "Hello, Flask and TensorFlow are working!"

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['image']
    image_path = "./uploaded_image.jpg"
    imagefile.save(image_path)

    img = Image.open(image_path).resize((128,128))
    img = np.array(img)/255.0
    img = img.reshape(1,128,128,3)

    pred = model.predict(img)
    result = np.argmax(pred)

    if result == 1:
        prediction = "Brain Tumor Detected"
    else:
        prediction = "No Brain Tumor Detected"

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
