import multiprocessing

barrier=multiprocessing.Barrier(11)
#finnish=multiprocessing.Barrier(10)

def f(i):
    while(True):
        barrier.wait()
        print(i)

prosesses=[]
for i in range(10):
    p=multiprocessing.Process(target=f, args=(i,))
    prosesses.append(p)
    prosesses[i].start()

while(True):
    asd=input("go")
    barrier.wait()

