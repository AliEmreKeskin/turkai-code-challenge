import multiprocessing
import cv2
import numpy as np
from datetime import datetime
import os
import sys

capture_barrier=multiprocessing.Barrier(3)
operation_barrier=multiprocessing.Barrier(3)

class Capture:

    def __init__(self,index,connection):
        # self.capture=cv2.VideoCapture(i)
        # _,self.frame=self.capture.read()
        self.process=multiprocessing.Process(target=self.handler, args=(connection, index))
        self.process.daemon=True
        self.process.start()

    def handler(self,conn,i):
        cap=cv2.VideoCapture(i)
        while(True):
            capture_barrier.wait()
            _,frame=cap.read()
            conn.send(frame)
            operation_barrier.wait()

    def __del__(self):
        pass
        # self.capture.release()


class Operation:
    def __init__(self,index0,index1):
        self.parent_conn_0,child_conn_0=multiprocessing.Pipe()
        self.parent_conn_1,child_conn_1=multiprocessing.Pipe()
        self.cap0=Capture(index0,child_conn_0)
        self.cap1=Capture(index1,child_conn_1)

    def run(self):
        print("b")
        while(True):
            capture_barrier.wait()
            operation_barrier.wait()
            frame0=self.parent_conn_0.recv()
            frame1=self.parent_conn_1.recv()
            cv2.imshow("frame 1",frame0)
            cv2.imshow("frame 2",frame1)

            key=cv2.waitKey(1)

            if key & 0xFF == ord('s'):
                cv2.imwrite("image_1.png",frame0)
                cv2.imwrite("image_2.png",frame1)

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