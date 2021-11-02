import numpy as np
import cv2
import time
from scipy import ndimage
import json, os, requests
#import Image
#from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from azure.storage.blob import BlockBlobService


#cascPath = 'haarcascade_frontalface_dataset.xml'  # dataset
#faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)  # 0 for web camera live stream
#  for cctv camera'rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
#  example of cctv or rtsp: 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=1_stream=0.sdp'

subscription_key = ""
assert subscription_key
face_api_url = 'https://faceapiforazure.cognitiveservices.azure.com/' + '/face/v1.0/detect'
headers = {'Ocp-Apim-Subscription-Key': subscription_key,'Content-Type': 'application/octet-stream','Cache-Control':'no-cache','pragma':'no-cache'}

params = {
    'detectionModel': 'detection_01',
    'returnFaceId': 'true'
}
gloabl_face_id = []

def camera_image():
    subscription_key = ""
    assert subscription_key
    face_api_url = 'https://faceapiforazure.cognitiveservices.azure.com/' + '/face/v1.0/detect'
    face_group_url = 'https://faceapiforazure.cognitiveservices.azure.com/' + '/face/v1.0/group'
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,'Content-Type': 'application/octet-stream','Cache-Control':'no-cache','pragma':'no-cache'}
    headers_grp = {'Ocp-Apim-Subscription-Key': subscription_key,'Content-Type': 'application/json','Cache-Control':'no-cache','pragma':'no-cache'}
    params = {'detectionModel': 'detection_01','returnFaceId': 'true'}
    threshold = 0.7
    sec = 0
    frameRate = 1.5
    account_key=""
    blob_service = BlockBlobService("facedetectionanton", account_key)
    nam = "test.jpg"
    #cam = cv2.VideoCapture('static/input_img.jpg',cv2.CAP_DSHOW)
    img=cv2.imread('newlatest/input_face.jpg')
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    url = "https://facedetectionanton.blob.core.windows.net/anton/"+nam
    count=0
    dic = {}
    sec = 0
    frameRate = 1.5
    count=0
    font = cv2.FONT_HERSHEY_SIMPLEX
	#success = getFrame(sec)
    #ret,img = cam.read()
    print("*** RANDOM ***")
    data = cv2.imwrite("newlatest/test_face.jpg",img)
    with open('newlatest/test_face.jpg', 'rb') as f:
        data = f.read()
    response = requests.post(face_api_url, params=params,headers=headers,data=data)
    img = np.array(img)
    result = response.json()
    print(result)
    person_count=0
    person_count=format(str(len(result)))
    print(person_count)
    cv2.putText(img,"Person detected {}".format(str(len(result))),(10,50), font, 0.75,(255,0,0),1,cv2.LINE_AA)
    for i in result:
        rect = i['faceRectangle']
        face_id = i['faceId']
	    #draw.rectangle(((rect['left'],rect['top']),(rect['left']+rect['width'],rect['top']+rect['height'])),outline='red')
        cv2.rectangle(img,(rect['left'],rect['top']),(rect['left']+rect['width'],rect['top']+rect['height']),(255,0,0),2)
        gloabl_face_id.append(face_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        data_grp = {}
        data_grp['faceIds']=gloabl_face_id
        response_grp = requests.post(face_group_url,headers=headers_grp,json=data_grp)
        if(len(gloabl_face_id)>2):
            grp_result = response_grp.json()
            count = len(grp_result['groups'])
            print(response_grp.json())
        print(response_grp)
        #cv2.putText(img,"Person detected {}".format(str(count)),(10,50), font, 0.75,(255,0,0),1,cv2.LINE_AA)
        #cv2.imwrite("static/new_img.jpg",img)
    #cv2.imshow("preview", img)
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    input_img=cv2.imread("newlatest/input_face.jpg")
    input_img = cv2.resize(input_img, (450, 250)) 
    cv2.imwrite("newlatest/input_face.jpg",input_img)

    cv2.imwrite("newlatest/output_face.jpg",img)
    return person_count
    #cv2.imshow("preview", img)
    
    #img.release()
    #cv2.destroyAllWindows()
        
    #return cv2.imencode('.jpg', img)[1].tobytes()
    
        #cv2.imshow('frame', img)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
#if __name__ == "__main__":
	#camera_image()
	
    
   
