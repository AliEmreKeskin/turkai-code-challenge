# Necessary packages
import multiprocessing
import cv2
import numpy as np
from datetime import datetime
import os
import sys
import signal
import imutils
import argparse
import time

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
    ap = argparse.ArgumentParser()
    ap.add_argument("-i0", "--index_camera_0", type=int, default=0, help="first camera index")
    ap.add_argument("-i1", "--index_camera_1", type=int, default=1, help="second camera index")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    ap.add_argument("-s", "--show", type=int, default=0, help="0=only yolo, 1=every thing")
    args = vars(ap.parse_args())
    sources=[args["index_camera_0"],args["index_camera_1"]]

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
        if(args["show"]==1):
            cv2.imshow("frame 1",curr_frame_1)
            cv2.imshow("frame 2",curr_frame_2)

        # Stitch frames
        images=[curr_frame_1,curr_frame_2]
        info("[ STITCHING ]")
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch(images)
        if(status==0):
            if(args["show"]==1):
                cv2.imshow("Stitched",stitched)

            # Yolo
            # Load the COCO class labels our YOLO model was trained on
            labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            # initialize a list of colors to represent each possible class label
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
            # derive the paths to the YOLO weights and model configuration
            weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
            configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])        
            # load our YOLO object detector trained on COCO dataset (80 classes)
            print("[INFO] loading YOLO from disk...")
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            # load our input image and grab its spatial dimensions
            image = stitched
            (H, W) = image.shape[:2]
            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()
            
            # show timing information on YOLO
            info("[INFO] YOLO took {:.6f} seconds".format(end - start))

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
            
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
            
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
            
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
            
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            # show the output image
            if(args["show"]==1 or args["show"]==0):
                cv2.imshow("Yolo", image)
        else:
            info("[ STITCHING FAILED WITH EROR CODE ({}) ]".format(status))

        

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