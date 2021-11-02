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
import cv2
import flask

import sys
import requests
import glob
import random
import pickle
import io
import tensorflow as tf
from scipy import ndimage
import time
#from camera import camera_stream
import base64
import json
#import pytesseract
from PIL import Image,ImageDraw

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

app = Flask(__name__, static_folder='newlatest')
#cors = CORS(app, resources={r"/foo": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'
#app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
CORS(app)

UPLOAD_FOLDER = 'newlatest'

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER



@cross_origin(origin='*',headers=['Content-Type','Authorization'])
@app.route('/human_run',methods=['POST','GET'])
def human():
    req = request.form
    print("worked")
    print(req)
    if request.method == 'POST':
        print("hi")
        isthisFile=request.files.get('filename')
        print(isthisFile)

        isthisFile.save("newlatest/"+'input_vid.mp4')
        
        class DetectorAPI:
            def __init__(self, path_to_ckpt):
                self.path_to_ckpt = path_to_ckpt

                self.detection_graph = tf.Graph()
                with self.detection_graph.as_default():
                    od_graph_def = tf.GraphDef()
                    with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')

                self.default_graph = self.detection_graph.as_default()
                self.sess = tf.Session(graph=self.detection_graph)

                # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            def processFrame(self, image):
                # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                start_time = time.time()
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})
                end_time = time.time()

                print("Elapsed Time:", end_time-start_time)

                im_height, im_width,_ = image.shape
                boxes_list = [None for i in range(boxes.shape[1])]
                for i in range(boxes.shape[1]):
                    boxes_list[i] = (int(boxes[0,i,0] * im_height),
                                int(boxes[0,i,1]*im_width),
                                int(boxes[0,i,2] * im_height),
                                int(boxes[0,i,3]*im_width))

                return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

            def close(self):
                self.sess.close()
                self.default_graph.close() 
        def gen_human():
            """Video streaming generator function."""
            def getFrame(sec):
                cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
                hasFrames,image = cap.read()
                return hasFrames
            model_path = 'faster_rcnn/frozen_inference_graph.pb'
            odapi = DetectorAPI(path_to_ckpt=model_path)
            threshold = 0.1
            cap = cv2.VideoCapture('newlatest/input_vid.mp4')
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
            print('run command...........................')
           
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('newlatest/output_vid.avi',fourcc, 20.0, (int(width), int(height)))
            

            # Read until video is completed
            while(True):
            # Capture frame-by-frame
                ret, img = cap.read()
                if ret == True:
                    img = cv2.resize(img, (int(width),int(height)))
                    #img = ndimage.rotate(img, 90)
                    boxes, scores, classes, num = odapi.processFrame(img)
                    final_score = np.squeeze(scores)    
                    count = 0
                    for i in range(len(boxes)):
                        if scores is None or final_score[i] > threshold:
                            count = count + 1
                        if classes[i] == 1 and scores[i] > threshold:
                            box = boxes[i]
                            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,"Person detected {}".format(str(count)),(10,50), font, 0.75,(255,0,0),1,cv2.LINE_AA)
                    #frame = cv2.imencode('.jpg', img)[1].tobytes()
                    #cv2.imwrite(img,"person1.jpg")
                    #cv2.imshow("image",img)
                    out.write(img)
                    print(count)
                    #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.1)
                    
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                if ret==False:
                    print("Video writing completed")
                    break
            cap.release()
            out.release()
            #cv2.destroyAllWindows()
            return "success"

        #return cv2.imencode('.jpg', img)[1].tobytes()
        #sec = sec + frameRate
        #sec = round(sec, 2)
        #success = getFrame(sec)
        #cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        #hasFrames,image = cap.read()
        #success=hasFrames
       # key = cv2.waitKey(1)
        #if key & 0xFF == ord('q'):
        #    break
    status= gen_human()
    if(status=='success'):
                print("sucess")
                import moviepy
                #from moviepy import VideoFileClip
                import moviepy.editor as moviepy
                import base64
                clip = moviepy.VideoFileClip("newlatest/output_vid.avi")
                clip.write_videofile("newlatest/output_vid.mp4")
                    
          
         
                with open("newlatest/output_vid.mp4", "rb") as videoFile1:
                # text = base64.b64encode(videoFile.read())
                    #text2=json.loads(text.decode('utf-8'))
                    #print(text)
                    # with open(file, 'rb') as fileObj:
                    image_data1 = videoFile1.read()
                    base64_data1 = base64.b64encode(image_data1)
                    fout1 = open("test2.txt", 'w')
                    fout1.write(str(base64_data1))
                    f2=base64_data1.decode()
                    fout1.close()
          
            
    
        # your api code
               
        # prepare the response: data
                response_data = {"text": 'Uploaded Successfully', "input_image": f2}
                print("----")
                #print(response_data)

               
                return flask.jsonify(response_data) 


    else:
            return 'error'

        




   



'''
@app.route('/vehicle_run',methods=['POST','GET'])
def vehicle():
    req = request.form
    print("worked")
    print(req)
    if request.method == 'POST':
        print("hi")
        isthisFile=request.files.get('filename')
        print(isthisFile)

        isthisFile.save("newlatest/"+'input_vid.mp4')
        import six.moves.urllib as urllib
        import sys
        import tarfile
        import zipfile
        import csv
        import time
        from packaging import version

        from collections import defaultdict
        from io import StringIO
        from matplotlib import pyplot as plt
        from PIL import Image

        # Object detection imports
        

        # initialize .csv
        with open('traffic_measurement.csv', 'w') as f:
            writer = csv.writer(f)
            csv_line = \
                'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
            writer.writerows([csv_line.split(',')])

        if version.parse(tf.__version__) < version.parse('1.4.0'):
            raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                            )

        # input video
        cap = cv2.VideoCapture('newlatest/input_vid.mp4')
        frame_width=int(cap.get(3))
        frame_height=int(cap.get(4))

        #FILE_OUTPUT="newlatest/output_vid.mp4"
        
        #Define the codec and create VideoWriter object. The output is stored in output.avi file
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #out=cv2.VideoWriter(FILE_OUTPUT,fourcc,10,(frame_width,frame_height))


       # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('newlatest/output_vid.avi',fourcc, 20.0,(frame_width,frame_height))

        # Variables
        total_passed_vehicle = 0  # using it to count vehicles

        # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
        # What model to download.
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = \
            'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

        # Download Model
        # uncomment if you have not download the model yet
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)


        # Helper code
        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape((im_height, im_width,
                    3)).astype(np.uint8)


        # Detection
        def object_detection_function():
            total_passed_vehicle = 0
            speed = 'waiting...'
            direction = 'waiting...'
            size = 'waiting...'
            color = 'waiting...'
            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:

                    # Definite input and output Tensors for detection_graph
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                    # Each box represents a part of the image where a particular object was detected.
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    # for all the frames that are extracted from input video
                    while cap.isOpened():
                        (ret, frame) = cap.read()

                        if not ret:
                            print ('end of the video file...')
                            break

                        input_frame = frame

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(input_frame, axis=0)

                        # Actual detection.
                        (boxes, scores, classes, num) = \
                            sess.run([detection_boxes, detection_scores,
                                    detection_classes, num_detections],
                                    feed_dict={image_tensor: image_np_expanded})

                        # Visualization of the results of a detection.
                        (counter, csv_line) = \
                            vis_util.visualize_boxes_and_labels_on_image_array(
                            cap.get(1),
                            input_frame,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=4,
                            )


                        #Save output in video file
                        out.write(input_frame)
                        total_passed_vehicle = total_passed_vehicle + counter

                        # insert information text to video frame
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            input_frame,
                            'Detected Vehicles: ' + str(total_passed_vehicle),
                            (10, 35),
                            font,
                            0.8,
                            (0, 0xFF, 0xFF),
                            2,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            )

                        # when the vehicle passed over line and counted, make the color of ROI line green
                        if counter == 1:
                            cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                        else:
                            cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)

                        # insert information text to video frame
                        cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                        cv2.putText(input_frame,'ROI Line',(545, 190),font,0.6,(0, 0, 0xFF),2,cv2.LINE_AA)
                        cv2.putText(input_frame,'LAST PASSED VEHICLE INFO',(11, 290),font,0.5,(0xFF, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(input_frame,'-Movement Direction: ' + direction,(14, 302),font,0.4,(0xFF, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                        cv2.putText(input_frame,'-Speed(km/h): ' + speed,(14, 312),font,0.4,(0xFF, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                        cv2.putText(input_frame,'-Color: ' + color,(14, 322),font,0.4,(0xFF, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                        cv2.putText(input_frame,'-Vehicle Size/Type: ' + size,(14, 332),font,0.4,(0xFF, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_COMPLEX_SMALL)

                        #cv2.imshow('vehicle detection', input_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        if csv_line != 'not_available':
                            with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)
                                (size, color, direction, speed) = \
                                    csv_line.split(',')
                                writer.writerows([csv_line.split(',')])
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()       
                    return 'success'
        status= object_detection_function()
        if(status=='success'):
                print("sucess")
                
                import moviepy.editor as moviepy
                clip = moviepy.VideoFileClip("newlatest/output_vid.avi")
                clip.write_videofile("newlatest/output_vid.mp4")
                
                import base64
                with open("newlatest/output_vid.mp4", "rb") as videoFile1:
                # text = base64.b64encode(videoFile.read())
                    #text2=json.loads(text.decode('utf-8'))
                    #print(text)
                    # with open(file, 'rb') as fileObj:
                    image_data1 = videoFile1.read()
                    base64_data1 = base64.b64encode(image_data1)
                    fout1 = open("test2.txt", 'w')
                    fout1.write(str(base64_data1))
                    f2=base64_data1.decode()
                    fout1.close()
          
            
    
        # your api code
               
        # prepare the response: data
                response_data = {"text": 'Uploaded Successfully', "input_image": f2}
                print("----")
                #print(response_data)

               
                return flask.jsonify(response_data) 

        else:
            return 'error'

'''

if __name__ == '__main__':
   app.run('0.0.0.0',port=5013)
