# ------------------------
#   USAGE
# ------------------------
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Load the image
img = cv2.imread(args["image"])
output = img.copy()

# Pre-process the image for classification
img = cv2.resize(img, (96, 96))
img = img.astype("float") / 255.0
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

# Load the trained convolutional neural network and the label binarizer
print("[INFO] Loading the network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# Classify the input image
print("[INFO] Classifying the image...")
proba = model.predict(img)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# We'll mark our prediction as "correct" of the input image filename contains the predicted label text
# (obviously this makes the assumption that you have named your testing image files this way)
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# Build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)