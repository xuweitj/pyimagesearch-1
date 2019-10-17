# ------------------------
#   USAGE
# ------------------------
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# ------------------------
#   IMPORTS
# ------------------------
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# Import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Initialize the number of epochs to train for, initial learning rate, batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# Initialize the data and labels
data = []
labels = []

# Grab the image paths and randomly shuffle them
print("[INFO] Loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# Loop over the input images
for imgPath in imagePaths:
    # Load the image, pre-process it and store it in the data list
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    img = img_to_array(img)
    data.append(img)
    # Extract the class label from the image path and update the labels first
    label = imgPath.split(os.path.sep)[-2]
    labels.append(label)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] Data Matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

# Binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Partition the data into training and testing splits using 80% for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Initialize the model
print("[INFO] Compiling the model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] Training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# Save the model to disk
print("[INFO] Serializing the network...")
model.save(args["model"])

# Save the label binarizer to disk
print("[INFO] Serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# Showing the history keys
print("[INFO] Showing history keys")
for key in H.history.keys():
    print(key)

# Plot the training loss and accuracy
print("[INFO] Training loss and accuracy...")
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])