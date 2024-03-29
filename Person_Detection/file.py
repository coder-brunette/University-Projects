import numpy as np
import tensorflow as tf
import cv2
import time

from scipy import ndimage

def process(path_to_ckpt_arg):
        path_to_ckpt = path_to_ckpt_arg

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        default_graph = detection_graph.as_default()
        sess = tf.Session(graph=detection_graph)

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def processFrame(image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
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

def close():
        sess.close()
        default_graph.close()
        
def getFrame(sec):
            cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = cap.read()
            return hasFrames

def main_content():
       
         model_path = 'faster_rcnn/frozen_inference_graph.pb'
         odapi = process(path_to_ckpt_arg=model_path)
         threshold = 0.7
         cap = cv2.VideoCapture('video4.mp4')
         sec = 0
         frameRate = 1.5
         #success = getFrame(sec)
         suc=cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
         hasFrames,image=cap.read()
         success=hasFrames
         while success:
            r, img = cap.read()
            try:
                img = cv2.resize(img, (480, 848))
                #img = cv2.resize(img, (1920, 1080))
                img = ndimage.rotate(img, 90)
            except Exception as e:
                print(str(e))
            boxes, scores, classes, num = processFrame(img)
            final_score = np.squeeze(scores)
            count = 0
            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if scores is None or final_score[i] > threshold:
                    count = count + 1
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,"Person detected {}".format(str(count)),(10,50), font, 0.75,(255,0,0),1,cv2.LINE_AA)
            cv2.imshow("preview", img)
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):

                break;
            


if __name__ == "__main__":
    main_content()

    
