# aicity2021
This project deals with the track 1(count the number of vehicles at the intersection) of AICITY challange 2021 
We have used DeepSORT on YOLOV3 to detect and track the number of vehicles the reason behind using it is along with distance and velocity features it also focus on appearance which makes it better than SORT previous popular algorithm for tracking
it is a three phase pipeline DTC(detection-tracking-counting)
step 1-To run this. you just need to clone this repo
step 2-To get the pre trained weights for YOLO models
use these links
https://pjreddie.com/media/files/yolov3.weights
https://pjreddie.com/media/files/yolov3-tiny.weights

And Execute convert.py to convert tensorflow weights and check for 4 new files in weights folder
one test video is provided in data/video/ by the name of cam_1 but you can also test it on your own data and get the result in data/video/result.avi
Execute demo.py file to see the results
For any help and advice contact CS20M009@iittp.ac.in
                                CS20M002@iittp.ac.in
          

#CVPR2021
