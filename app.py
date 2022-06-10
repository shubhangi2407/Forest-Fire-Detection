from __future__ import division, print_function
import imp
import os
import sys
import glob
import re
import numpy as np

#Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

#Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Defining flask applications
app = Flask(__name__)

# model path
MODEL_PATH = 'E:/Combined Alzheimer/vgg.h5'
model = load_model(MODEL_PATH)
model._make_predict_function()
print("Model loaded ... Check http://127.0.0.1:5000/")

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x  = image.img_to_array(img)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path  = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)


