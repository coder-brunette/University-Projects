from flask import Flask, render_template, request, Response, send_file
from flask_cors import CORS, cross_origin 
from flask_restful import Resource, Api
from json import dumps
from werkzeug.utils import secure_filename
from flask_jsonpify import jsonify
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import json
import numpy
#from darkflow.net.build import TFNet
import cv2
from inference_Cap import bottle_cap_model
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import io
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import flask
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


app = Flask(__name__, static_folder='newlatest')
#cors = CORS(app, resources={r"/foo": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'
#app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
CORS(app)

UPLOAD_FOLDER = 'newlatest'

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER



@cross_origin(origin='*',headers=['Content-Type','Authorization'])
@app.route('/',methods=['POST','GET'])
def bottle_page():
    return render_template('bottle_app.html')

@app.route('/bottle_cap',methods=['POST','GET'])
def face():
    req = request.form
    print("worked")
    print(req)
    if request.method == 'POST':
        print("hi")
        #isthisFile=request.form['filename']
        isthisFile=request.files.get('filename')
        
        print(isthisFile)

        isthisFile.save("newlatest/"+'download.jpg')

        cam=bottle_cap_model()

        if (cam) :
                print("exex")
               
                import base64

                def get_encoded_img(image_path):
                        img = Image.open(image_path, mode='r')
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG')
                        my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
                        return my_encoded_img
                '''
                with open("testvideo.mp4", "rb") as videoFile:
                # text = base64.b64encode(videoFile.read())
                    #text2=json.loads(text.decode('utf-8'))
                    #print(text)
                    # with open(file, 'rb') as fileObj:
                    image_data = videoFile.read()
                    base64_data = base64.b64encode(image_data)
                    fout = open("test.txt", 'w')
                    fout.write(str(base64_data))
                    f=base64_data.decode()
                    fout.close()
        '''
        
        # your api code
                msg='hi'
                imgpath1 = 'newlatest/download.jpg'
                imgpath2='newlatest/im1.jpg'
                img1 = get_encoded_img(imgpath1)
                img2 = get_encoded_img(imgpath2)
        # prepare the response: data
                response_data = {"text": 'Uploaded Successfully', "person_count": cam, "input_image": img1,"output_image":img2}
                print("----")
                #print(response_data)
                return flask.jsonify(response_data ) 

    
    #return "worked"


if __name__ == '__main__':

    #app.run(host='0.0.0.0', port=5002)
    app.run(host='0.0.0.0', port=5000)


