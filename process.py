from multiprocessing import Process
import os
from datetime import datetime


def info(title):
    print(title)
    dt = datetime.now()
    print(dt.microsecond/1000)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p0 = Process(target=f, args=('bob',))
    p1 = Process(target=f, args=('emre',))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
