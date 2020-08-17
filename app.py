import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')  # 强制使用 cpu

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'
LABEL_PATH = 'models/labels.txt'

# Load your own trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')
labels = [x.strip() for x in open(LABEL_PATH, 'rt').readlines()]
image_size = (256, 256)


def model_predict(img, model):
    img = img.resize(image_size)
    x = preprocess_input(img)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    idx = np.argmax(preds)
    return float(preds[0][idx]), labels[idx]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        pred_proba, pred_class = model_predict(img, model)

        # Serialize the result, you can add additional fields
        return jsonify(result=pred_class, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
