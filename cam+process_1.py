import numpy as np
import cv2
from multiprocessing import Process
#import os
from datetime import datetime
import sys

def info(title):
    print(title)
    dt = datetime.now()
    print(dt.hour,dt.minute,dt.second,dt.microsecond/1000)
    print('module name:', __name__)
    #print('parent process:', os.getppid())
    #print('process id:', os.getpid())

def f(cap,filename):
    info('function f**************')
    print(filename)
    ret,frame=cap.read()
    cv2.imwrite(filename,frame)

if __name__ == '__main__':

    if(len(sys.argv)<2):
        cap0=cv2.VideoCapture(0)
        cap1=cv2.VideoCapture(1)
    else:
        i0=int(sys.argv[1])
        i1=int(sys.argv[2])
        cap0=cv2.VideoCapture(i0)
        cap1=cv2.VideoCapture(i1)

        


    info('main line***************')
    
    p0 = Process(target=f, args=(cap0,"0.jpg",))
    p1 = Process(target=f, args=(cap1,"1.jpg",))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    cap0.release()
    cap1.release()
