# ------------------------
#   USAGE
# ------------------------
# python predict-video.py --model model/activity.model --label-bin model/lb.pickle \
# --input example_clips/lifting.mp4 --output output/lifting_128avg.avi --size 128

# ------------------------
#   IMPORTS
# ------------------------
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True, help="path to  label binarizer")
ap.add_argument("-i", "--input", required=True, help="path to our input video")
ap.add_argument("-o", "--output", required=True, help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128, help="size of queue for averaging")
args = vars(ap.parse_args())

# Load the trained model and label binarizer from the disk
print("[INFO] Loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# Initialize the image man for mean subtraction along with the predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# Initialize the video stream, pointer to output video file and frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed then we have reached the end of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # clone the output frame then convert it from BGR to RGB ordering, resize the frame to a fixed 224x224 size
    # then perform mean subtraction
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean
    # make predictions on the frame and then update the predictions queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    # perform prediction averaging over the current history of previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    # draw the activity on the output frame
    text = "activity: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    # write the output frame to disk
    writer.write(output)
    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' was pressed, break from the loop
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] Cleaning up...")
writer.release()
vs.release()