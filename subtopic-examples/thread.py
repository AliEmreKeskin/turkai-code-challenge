import threading
import cv2
import numpy as np
from datetime import datetime
import os
import sys

capture_barrier=threading.Barrier(3)
operation_barrier=threading.Barrier(3)

class Capture:

    def __init__(self,index):
        self.index=index
        self.capture=cv2.VideoCapture(index)
        _,self.frame=self.capture.read()
        self.process=threading.Thread(target=self.handler, args=())
        self.process.daemon=True
        self.process.start()

    def handler(self):
        while(True):
            capture_barrier.wait()
            dt = datetime.now()
            print(self.index,dt.hour,dt.minute,dt.second,dt.microsecond/1000)
            _,self.frame=self.capture.read()
            dt = datetime.now()
            print(self.index,dt.hour,dt.minute,dt.second,dt.microsecond/1000)
            operation_barrier.wait()

    def __del__(self):
        self.capture.release()


class Operation:
    def __init__(self,index0,index1):
        self.cap0=Capture(index0)
        self.cap1=Capture(index1)

    def run(self):
        while(True):
            capture_barrier.wait()
            operation_barrier.wait()
            cv2.imshow("frame 1",self.cap0.frame)
            cv2.imshow("frame 2",self.cap1.frame)

            key=cv2.waitKey(1)

            if key & 0xFF == ord('s'):
                cv2.imwrite("image_1.png",self.cap0.frame)
                cv2.imwrite("image_2.png",self.cap1.frame)

            if key & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    i0=0
    i1=1
    if(len(sys.argv)<2):
        pass
    else:
        i0=int(sys.argv[1])
        i1=int(sys.argv[2])
    
    op=Operation(i0,i1)
    op.run()