from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
from shapely.geometry import Point, Polygon
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')
df_results=pd.DataFrame()
df_tracks=pd.DataFrame()
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)
#giving data to execute 
f = open("list_video_id.txt", "r")
content=f.readlines()
f.close()
video_list=[]
index_list=[]
for line in content:
    splitline=line.split(" ")
    # print(splitline)
    video_list.append(splitline[1].strip())
for i in video_list:
    vid = cv2.VideoCapture('./data/video/'+i)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
    vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #saving results in data->video->results
    out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

    counter = []
    frame_id=0
    


    while True:
        _, img = vid.read()
        frame_id+=1
        
        if img is None:
            print('Completed')
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)
        color=(0,255,0)
        thickness=5
        isClosed=True
        pts=np.array([[1,150],[844,96],[1277,277],[1277,750],[2,683]],np.int32)
        poly=Polygon(pts)
#drawing region of interest polygon
        roi=cv2.polylines(img,[pts],isClosed,color,thickness)

        t1 = time.time()
#getting boxes ,confidence scores , classes
        boxes, scores, classes, nums = yolo.predict(img_in)
        

        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                    zip(converted_boxes, scores[0], names, features)]
        
        # tlwh=list(d.tlwh for d in detections)
        boxs = np.array([d.tlwh for d in detections]) #top left coordinates,width,hieght
        scores = np.array([d.confidence for d in detections]) #confidence score
        classes = np.array([d.class_name for d in detections]) #classes
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        df_results=df_results.append({
            'tlwh':boxs,
            'frame_id':frame_id,
            'confidence':scores

        },ignore_index=True)
        # print(df_results.head(20))

        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

        current_count = int(0)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >1:
                continue
            #get bounding box , class name and color to each box
            bbox = track.to_tlbr()
            class_name= track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            df_tracks=df_tracks.append({
                'vehicle_id':track.track_id,
                'bbox_coordinates':bbox,
                'class_name':class_name,
            },ignore_index=True)



            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                        +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                        (255, 255, 255), 2)
            # center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
            # pts[track.track_id].append(center)

            # for j in range(1, len(pts[track.track_id])):
            #     if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
            #         continue
            #     thickness = int(np.sqrt(64/float(j+1))*2)
            #     cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

            height, width, _ = img.shape
            
        # cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
        # cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)
        
            center_y = int(((bbox[1])+(bbox[3]))/2)
            center_x = int(((bbox[0])+(bbox[2]))/2)

            p1=Point(center_x,center_y)

            if p1.within(poly):
                if class_name == 'car' or class_name == 'truck':
                    counter.append(int(track.track_id))
                    current_count += 1

        total_count = len(set(counter))
        cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
        cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

        fps = 1./(time.time()-t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
        cv2.resizeWindow('output', 1024, 768)
        cv2.imshow('output', img)
        out.write(img)
        df_results.to_csv('/Users/lakshaykalra/Desktop/results.csv',sep=',',encoding='utf-8',index=False)
        df_tracks.to_csv('/Users/lakshaykalra/Desktop/tracks.csv',sep=',',encoding='utf-8',index=False)

        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    



