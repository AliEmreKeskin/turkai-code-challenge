import multiprocessing

capture_barrier=multiprocessing.Barrier(11)
operation_barrier=multiprocessing.Barrier(11)

def f(i):
    while(True):
        capture_barrier.wait()
        print(i)
        operation_barrier.wait()

prosesses=[]
for i in range(10):
    p=multiprocessing.Process(target=f, args=(i,))
    prosesses.append(p)
    prosesses[i].start()

while(True):
    capture_barrier.wait()
    operation_barrier.wait()
    print("***********")

