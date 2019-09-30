# ------------------------
#   USAGE
# ------------------------
# python multi-obj-track-fast.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
# 	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video input/race.mp4 \
# 	--label person --output output/race_output_fast.avi

# ------------------------
#   IMPORTS
# ------------------------
from imutils.video import FPS
import multiprocessing
import time
import os
import numpy as np
import argparse
import imutils
import dlib
import cv2


# ------------------------
#   FUNCTIONS
# ------------------------
def start_tracker(box, label, rgb, input_queue, output_queue):
    # Construct a dlib rectangle object from the bounding box coordinates and start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)
    # Loop indefinitely -- this function will be called as a daemon process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        rgb = input_queue.get()
        # if there was an entry in our queue, process it
        if rgb is not None:
            # update the tracker and grab the position of the tracked object
            t.update(rgb)
            pos = t.get_position()
            # unpack the position object
            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())
            # add the label + bounding box coordinates to the output queue
            output_queue.put((label, (start_x, start_y, end_x, end_y)))


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-l", "--label", required=True, help="class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize the list of queues -- both input queue and output queue for *every* object that we will be tracking
input_queues = []
output_queues = []

# Initialize the list of class labels that the MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
		   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Load our serialized model from disk
print("[INFO] Loading the model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream and the output video writer.
print("[INFO] Starting the video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# Start the frames per second throughput estimator
fps = FPS().start()
record = True
refresh = False

# Loop over the frames from the video file stream
while True:
    # grab the next frame from the video file
    if record:
        (grabbed, frame) = vs.read()
    else:
        (grabbed, frame) = (grabbed, frame)
    # check to see if we have reached the end of the video file
    if frame is None:
        break
    # resize the frame for faster processing and then convert the frame from BGR to RGB ordering because of dlib
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # if we are supposed to be writing a video to disk initialize the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # if the list of queues is empty then we know we have yet to create our first object tracker
    if len(input_queues) == 0:
        # grab the frame dimensions and convert the frame to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by requiring a minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the detections list
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                # if the class label is not a person ignore it
                if CLASSES[idx] != "person":
                    continue
                # compute the (x,y) coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)
                # create two brand new input and output queues respectively
                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()
                input_queues.append(iq)
                output_queues.append(oq)
                # spawn a daemon process for a new object tracker
                p = multiprocessing.Process(target=start_tracker, args=(bb, label, rgb, iq, oq))
                p.daemon = True
                print("Process Start")
                p.start()
                # key options
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    refresh = True
                # refresh process
                if refresh:
                    # stop daemon process
                    print("End Process")
                    p.daemon = False
                    p.terminate()
                    time.sleep(1)
                    # spawn a new daemon process for a new object tracker
                    p = multiprocessing.Process(target=start_tracker, args=(bb, label, rgb, iq, oq))
                    p.daemon = True
                    p.start()
                    refresh = False # reset refresh flag
                    print("Refresh: ", refresh)
                # grab the corresponding class label for the detection and draw the bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # otherwise, we've already performed the detection so let's track multiple objects
    else:
        # loop over each of our input queues and add the input RGB frame to it, enabling us to update each of the
        # respective object trackers running in separate processes
        for iq in input_queues:
            iq.put(rgb)
        # loop over each of the output queues
        for oq in output_queues:
            # grab the updated bounding box coordinates for the object
            # -- the .get method is a blocking operation so this will pause our execution until the respective
            # process finishes the tracking update
            (label, (startX, startY, endX, endY)) = oq.get()
            # draw the bounding box from the correlation object tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed break from the loop
    if key == ord("q"):
        break
    # if the 'p' key is pressed pause the loop
    elif key == ord("p"):
        record = False
        print("Record: ", record)
    # if the 's' key is pressed take screenshot
    elif key == ord("s"):
        record = False
        os.system("import screenshot.png")
    # if the 'c' key is pressed continue the loop
    elif key == ord("c"):
        record = True
        print("Record: ", record)
    # if the 'r' key is pressed refresh loop process
    elif key == ord("r"):
        refresh = True
        print("Refresh: ", refresh)
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()



