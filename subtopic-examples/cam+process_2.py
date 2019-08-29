import numpy as np
import cv2
from multiprocessing import Process
#import os
from datetime import datetime

cap0=cv2.VideoCapture(0)
cap1=cv2.VideoCapture(1)
cap=[cap0,cap1]
_,frame0=cap0.read()
_,frame1=cap1.read()
frame=[frame0,frame1]

def info(title):
    print(title)
    dt = datetime.now()
    print(dt.hour,dt.minute,dt.second,dt.microsecond/1000)
    print('module name:', __name__)
    #print('parent process:', os.getppid())
    #print('process id:', os.getpid())

def f(i):
    info('function f**************')
    ret,frame[i]=cap[i].read()

if __name__ == '__main__':
    info('main line***************')

    p0 = Process(target=f, args=(0,))
    p1 = Process(target=f, args=(1,))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    cv2.imwrite("0.jpg",frame[0])
    cv2.imwrite("1.jpg",frame[1])
