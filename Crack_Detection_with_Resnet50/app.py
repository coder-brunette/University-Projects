
from flask import Flask, render_template, Response,request, redirect, url_for
import cv2
import numpy as np
import glob
import random
import pickle
import os
#from testing_code import crack
#from frcnn_crack import predictRoute


#from keras.models import load_model
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions
#from src import VG16
from flask_cors import CORS, cross_origin 
import io
from PIL import Image
from resnet import start_resnet
from flask import Flask
import flask

from matplotlib import pyplot as plt

import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='newlatest')
CORS(app)
app.config['UPLOAD_PATH'] = 'newlatest'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/",methods=['GET'])
def test():
    return "API IS UP"


@app.route('/resnet', methods = ['POST','GET'])
def resnet():
    req = request.form
    print("worked")
    print(req)
    if request.method == 'POST':
        print("hi")
        isthisFile=request.files.get('filename')
        print(isthisFile)

        isthisFile.save("newlatest/"+'input_segment.jpg')

        resnet_inp=start_resnet()
        if(resnet_inp=="success"):
            print("sucess")
                
            import base64
            def get_encoded_img(image_path):
                        img = Image.open(image_path, mode='r')
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG')
                        my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
                        return my_encoded_img
        # your api code
        imgpath1 = 'newlatest/input_segment.jpg'
        imgpath2='newlatest/pic.jpg'
        img1 = get_encoded_img(imgpath1)
        img2 = get_encoded_img(imgpath2)
        # prepare the response: data
        response_data = {"text": 'Uploaded Successfully', "input_image": img1,"output_image":img2}
        print("----")
                #print(response_data)
        return flask.jsonify(response_data ) 

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

if __name__ == '__main__':
     app.run(host ='0.0.0.0', port = 5009, debug = True)



#Need to do Testcase Validations
#1.Test Flask app and packages using pip freeze
#2.Test JSON Requests and Responses
#3.Validate Exceptions