# -----------------------------
#   USAGE
# -----------------------------
# python train.py --plot images/cifar10_adam.png --optimizer adam
# python train.py --plot images/cifar10_rectified_adam.png --optimizer radam

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# -----------------------------
#   IMPORTS
# -----------------------------
from pyimagesearch.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras_radam import RAdam
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, required=True, help="path to output training plot")
ap.add_argument("-o", "--optimizer", type=str, default="adam", choices=["adam", "radam"], help="type of optmizer")
args = vars(ap.parse_args())

# Initialize the number of epochs to train for and batch size
EPOCHS = 75
BS = 128

# Load the training and testing data, then scale it into the range [0, 1]
print("[INFO] Loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Check if we are using Adam
if args["optimizer"] == "adam":
    # initialize the Adam optimizer
    print("[INFO] Using the Adam optimizer")
    opt = Adam(lr=1e-3)
# Otherwise, use the Rectified Adam
else:
    # initialize the Rectified Adam optimizer
    print("[INFO] Using the Rectified Adam optimizer")
    opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)

# Initialize our optimizer and model, then compile it
model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] Training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=trainX.shape[0] // BS, epochs=EPOCHS, verbose=1)

# Evaluate the network
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Determine the number of epochs and then construct the plot title
N = np.arange(0, EPOCHS)
title = "Training Loss and Accuracy on CIFAR-10 ({})".format(args["optimizer"])

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title(title)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])