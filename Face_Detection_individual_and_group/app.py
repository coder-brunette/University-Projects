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

#from shape_identify import shape_stream
from camera_face import camera_image
#from new_detect import soc_detect
import flask
#import azure.cognitiveservices.speech as speechsdk
#from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
#from azure.cognitiveservices.speech.audio import AudioOutputConfig
import sys
import requests
import glob
import random
import pickle
import io
#from frcnn_crack import predictRoute
#from pdfgenerator import pf
#from new_detect import soc_detect
import time
#from camera import camera_stream
import base64
import json
#import pytesseract
from PIL import Image,ImageDraw
#from new_detect import soc_detect

app = Flask(__name__, static_folder='newlatest')
#cors = CORS(app, resources={r"/foo": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'
#app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
CORS(app)

UPLOAD_FOLDER = 'newlatest'

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER



@cross_origin(origin='*',headers=['Content-Type','Authorization'])


@app.route('/face_run',methods=['POST','GET'])
def face():
    req = request.form
    print("worked")
    print(req)
    if request.method == 'POST':
        print("hi")
        #isthisFile=request.form['filename']
        isthisFile=request.files.get('filename')
        
        print(isthisFile)

        isthisFile.save("newlatest/"+'input_face.jpg')

        cam=camera_image()

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
                imgpath1 = 'newlatest/input_face.jpg'
                imgpath2='newlatest/output_face.jpg'
                img1 = get_encoded_img(imgpath1)
                img2 = get_encoded_img(imgpath2)
        # prepare the response: data
                response_data = {"text": 'Uploaded Successfully', "person_count": cam, "input_image": img1,"output_image":img2}
                print("----")
                #print(response_data)
                return flask.jsonify(response_data ) 

    
    #return "worked"

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5002)
   
