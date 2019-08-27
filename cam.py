import numpy as np
import cv2
import sys

if(len(sys.argv)<2):
    cap = cv2.VideoCapture(0)
else:
    i=int(sys.argv[1])
    cap = cv2.VideoCapture(i)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)

    key=cv2.waitKey(1)

    if key & 0xFF == ord('s'):
	    cv2.imwrite("image.png",frame)

    if key & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
