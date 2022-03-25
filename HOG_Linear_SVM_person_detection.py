from __future__ import print_function # to ensure our code run normaly in python 2.7 and 3
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

hog = cv2.HOGDescriptor() # Initialize HOG
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # Set SVM to be pretrained on  pedestrian detector

# Read the video
cap = cv2.VideoCapture("video/video4.mp4")

# Get frame width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
# define the frame size variable
size = (width, height)
# Set video saving parameters
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video_4.avi', fourcc, 20.0, size)

while True:
    ret, frame = cap.read()
    
    if frame is None:
        break
        
    frame = cv2.resize(frame, (width, height))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect people in the image
    (boxes, weights) = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.1)
    
    # Loop through the original bounding boxes
    for (x,y,w,h) in boxes:
        
        # Application of non-maxima suppression to get accurate bounidng boxes 
        boxes = np.array([[x,y,x+w, y+h] for (x,y,w, h) in boxes])
        # Apply non-maxima suppression
        NMS = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        
        for (xA, yA, xB, yB) in NMS:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0,244,0), 2)
            
        # Write the output video 
        #out.write(frame.astype('uint8'))
        out.write(frame)
        cv2.imshow("Video Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
out.release()
# finally, close the window
cv2.destroyAllWindows()