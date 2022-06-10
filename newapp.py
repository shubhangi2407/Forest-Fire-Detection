import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


model =load_model('D:/Vinu project/fire.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return 'Fire'
    elif classNo == 1:
        return 'Non Fire'


def getPreprocessedimage(img):
    IMG_HE = 224
    IMG_WI = 224
    new_img=cv2.imread(img)
  
    new_img=cv2.resize(new_img, (IMG_HE, IMG_WI))
    new_img = new_img.reshape(1,224,224,3)
    new_img=np.array(new_img)
    new_img = new_img/255
    
    return new_img



def getResult(img):
    image=getPreprocessedimage(img)
    result= model.predict(image)

    if result > 0.5:
        result = 1
    else:
        result = 0
    #pred = (result > 0.5)
    #pred=np.argmax(result,axis=1)
    return result


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (224,224))
    x = np.array(img)
    preds = model.predict(x)
    pred=np.argmax(preds,axis=1)
    return pred[0]



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #preds = model_predict(file_path, model) 

        value=getResult(file_path)
        result=get_className(value) 

    return result
    print(result)
#return None


if __name__ == '__main__':
    app.run(debug=True)