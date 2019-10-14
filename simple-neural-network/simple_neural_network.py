# ------------------------
#   USAGE
# ------------------------
# python simple_neural_network.py --dataset kaggle_dogs_vs_cats --model output/simple_neural_network.hdf5

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os


# ------------------------
#   FUNCTIONS
# ------------------------
def image_to_feature_vector(img, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
    return cv2.resize(img, size).flatten()


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model file")
args = vars(ap.parse_args())

# Grab the list of images that we'll be describing
print("[INFO] Describing the images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize the data matrix and labels list
data = []
labels = []

# Loop over the input images
for (i, imgPath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that the path with the
    # format:/path/to/dataset/{class}.{image_num}.jpg
    img = cv2.imread(imgPath)
    label = imgPath.split(os.path.sep)[-1].split(".")[0]
    # construct the feature vector raw pixel intensities, then update the data matrix and labels first
    features = image_to_feature_vector(img)
    data.append(features)
    labels.append(label)
    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] Processed {}/{}".format(i, len(imagePaths)))

# Encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Scale the input image pixels to the range [0, 1], then transform the labels into vectors in the range [0, num_classes]
# -- this generates a vector for each label where the index of the label is set to '1' and all other entries to '0'
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)

# Partition the data intro training and testing splits, using 75% for training and 25% for testing
print("[INFO] Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

# Training the model using SGD
print("[INFO] Compiling the model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)

# Show the accuracy on the testing set
print("[INFO] Evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

# Dump the network architecture and weights to file
print("[INFO] Dumping architecture and weights to file...")
model.save(args["model"])
