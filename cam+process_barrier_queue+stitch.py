# Necessary packages
import multiprocessing
import cv2
import numpy as np
from datetime import datetime
import os
import sys
import signal
import imutils

# Barrier to start capturing processes
capture_barrier=multiprocessing.Barrier(3)

# Barrier to start processing the gathered frames
operation_barrier=multiprocessing.Barrier(3)

# Lock for keeping information outputs mutually exclusive
info_lock=multiprocessing.Lock()

# Function for outputing some information to console atomically
def info(title):
    info_lock.acquire()
    print(title)
    dt = datetime.now()
    print(dt.hour,dt.minute,dt.second,dt.microsecond/1000)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    info_lock.release()

# Function to capture frames from parallel processes
def capture(source,queue):
    camera=cv2.VideoCapture(source)
    while(True):
        capture_barrier.wait()
        
        info("[ CAPTURE START ({}) ]".format(source))
        ret,frame=camera.read()
        info("[ CAPTURE FINISH ({}) ]".format(source))

        queue.put(frame)
        operation_barrier.wait()

# Main function for parent process
if __name__ == '__main__':

    # Manage arguments
    if(len(sys.argv)<2):
        sources=[0,1]
    else:
        i0=int(sys.argv[1])
        i1=int(sys.argv[2])
        sources=[i0,i1]

    # Queue for interprocess communication
    # Used for sharing frames between processes
    frame_queue=multiprocessing.Queue()

    # Processes for capturing frames
    p0=multiprocessing.Process(target=capture,args=(sources[0],frame_queue,))
    p1=multiprocessing.Process(target=capture,args=(sources[1],frame_queue,))
    p0.start()
    p1.start()

    # Loop for main operation
    while(True):

        # Gather synchronised frames
        capture_barrier.wait()
        operation_barrier.wait()
        info("[ GET IMAGES ]")
        curr_frame_1=frame_queue.get()
        curr_frame_2=frame_queue.get()
        cv2.imshow("frame 1",curr_frame_1)
        cv2.imshow("frame 2",curr_frame_2)

        # Stitch frames
        images=[curr_frame_1,curr_frame_2]
        info("[ STITCHING ]")
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch(images)
        if(status==0):
            cv2.imshow("Stitched",stitched)
        else:
            info("[ STITCHING FAILED WITH EROR CODE ({}) ]".format(status))


        # Yolo

        # Keyboard controls
        key=cv2.waitKey(1)
        # Save
        if key & 0xFF == ord('s'):
            cv2.imwrite("image_1.png",curr_frame_1)
            cv2.imwrite("image_2.png",curr_frame_2)
        # Quit
        if key & 0xFF == ord('q'):
            # Kill capturing processes so release camera and other resources
            os.kill(p0.pid, signal.SIGKILL)
            os.kill(p1.pid, signal.SIGKILL)
            break

# Close OpenCV windows
cv2.destroyAllWindows()