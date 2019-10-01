# ------------------------
#   USAGE
# ------------------------
# python detect-faces.py --images images/rooster.jpg --prototxt deploy.prototxt.txt \
# --model res10_300x300_ssd_iter_140000.caffemodel

# ------------------------
#   IMPORT
# ------------------------
# import the necessary packages
import numpy as np
import argparse
import cv2

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load the input image and construct an input blob for the image by resizing to a fixed 300x300 size and normalize it
img = cv2.imread(args["image"])
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the detections and predictions
print("[INFO] Computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e, probability) associated with the prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x,y) coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
