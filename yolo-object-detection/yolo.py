# ------------------------
#   USAGE
# ------------------------
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels that our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize the color list to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO]  Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load the input image and grab its spatial dimensions
img = cv2.imread(args["image"])
(H, W) = img.shape[:2]

# Determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Construct a blob from the input image, perform a forward pass of the YOLO object detector and that will give us
# bounding boxes alongside its associated probabilities
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# Show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# Initialize the list of detected bounding boxes, confidences and class IDs respectively
boxes = []
confidences = []
classIDs = []

# Loop over each one of the layer outputs
for output in layerOutputs:
    # loop over each one of the detections
    for detection in output:
        # extract the class ID and confidence (i.e, probability) of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # filter out weak predictions by ensuring the detected probabilityy is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the size of the image, keepin in mind that YOLO
            # actually returns the center (x,y) coordinates of the bounding box followed by the boxes width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x,y) coordinates to derive the top and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update the list of bounding box coordinates, confidences and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# Ensure at least on detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)