# ------------------------
#   USAGE
# ------------------------
# python deep-learning-with-opencv.py --image images/jemma.png --prototxt bvlc_googlenet.prototxt
# --model bvlc_googlenet.caffemodel --labels synset_words.txt

# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import argparse
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# Load the input image from disk
img = cv2.imread(args["image"])

# Load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# The CNN requires fixed spatial dimensions for our input images so we need to ensure it is resized to 224x224 pixels
# while performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command the "blob" now has the shape (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))

# Load the serialized model from disk
print("[INFO] Loading serialized model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Set the blob as input to the network and perform a forward-pass to obtain the output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] The classification process took approximately {:.5} seconds".format(end - start))

# Sort the indexes of the probabilities in descending order (higher probability first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1][:5]

# Loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
        cv2.putText(img, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # display the predicted label + associated probability to the console
    print("[INFO] {}. Label: {}, Probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))

# Display the output image
cv2.imshow("Image", img)
cv2.waitKey(0)