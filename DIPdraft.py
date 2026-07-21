# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 08:14:55 2026

@author: Jesslyn Wong
Student ID: 23029804
"""

import cv2
import numpy as np

# 1. read video
video = cv2.VideoCapture("CSC 2014_Group Assignment_April Sem 2026/street.mp4")

# 2. 
out = cv2.VideoWriter('processed_video.avi',    # Set the file name of the new video
                      cv2.VideoWriter_fourcc(*'MJPG'),  # Set the codec
                      30.0,     # Set the frame rate
                      (1280, 720)   # set the resolution (width, height)
                      )


# 3. get total number of the frames
total_num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

# 4. to loop through all the frames
for frame_count in range(0, int(total_num_frames)):
    success, frame = video.read()   # read a single frame from the video
       
    
    if not success:
        break
    
    # Do something here
    # nrow, ncol, nchannel = frame.shape
    
    # mask = np.ones((nrow, ncol, nchannel), dtype=np.uint8)
    # mask[0:nrow//2, 0:ncol//2, :] = 0
    
    # frame = frame * mask
    
   
    print(frame_count)
    
    # blur face
    face_cascade = cv2.CascadeClassifier("CSC 2014_Group Assignment_April Sem 2026/face_detector.xml")   # load pretrained Haar cascade model
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)    # perform face detection
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0,0),2)
        
        cv2.imshow("New frame", frame)
        cv2.waitKey()
    
    # out.write(frame)    # save processed frame into the new video
    
cv2.destroyAllWindows()

print("hi")
print("bye")
print("HEWEEOOO IM CINDYYY")