import numpy as np
import cv2
import sys
from datetime import datetime

if(len(sys.argv)<2):
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
else:
    i0=int(sys.argv[1])
    i1=int(sys.argv[2])
    cap0 = cv2.VideoCapture(i0)
    cap1 = cv2.VideoCapture(i1)

while(True):
    # Capture frame-by-frame
    print("*********")
    dt = datetime.now()
    print(dt.hour,dt.minute,dt.second,dt.microsecond/1000)
    ret, frame0 = cap0.read()
    dt = datetime.now()
    print(dt.hour,dt.minute,dt.second,dt.microsecond/1000)
    ret, frame1 = cap1.read()
    dt = datetime.now()
    print(dt.hour,dt.minute,dt.second,dt.microsecond/1000)

    # Display the resulting frame
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)

    key=cv2.waitKey(1)

    if key & 0xFF == ord('s'):
        cv2.imwrite("image0.png",frame0)
        cv2.imwrite("image1.png",frame1)

    if key & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()
