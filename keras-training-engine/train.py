# ------------------------
#   USAGE
# ------------------------
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints \
# 	--model output/checkpoints/epoch_40.hdf5 --start-epoch 40
# python train.py --checkpoints output/checkpoints \
#       --model output/checkpoints/epoch_50.hdf5 --start-epoch 50

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# ------------------------
#   IMPORT
# ------------------------
# import the necessary packages
from pyimagesearch.callbacks.epoch_check_point import EpochCheckPoint
from pyimagesearch.callbacks.training_monitor import TrainingMonitor
from pyimagesearch.nn.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import cv2
import sys
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# Grab the Fashion MNIST dataset (if this is your first time running this the dataset will be automatically downloaded)
print("[INFO] Loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# Fashion MNIST images are 28x28 but the network we will be training is expecting 32x32 images
trainX = np.array([cv2.resize(x, (32, 32)) for x in trainX])
testX = np.array([cv2.resize(x, (32, 32)) for x in testX])

# Scale data to the range of [0,1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Reshape the data matrices to include a channel dimension (required for training)
trainX = trainX.reshape((trainX.shape[0], 32, 32, 1))
testX = testX.reshape((testX.shape[0], 32, 32, 1))

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

# If there is no specific model checkpoint supplied, then initialize the network (ResNet-56) and compile the model
if args["model"] is None:
    print("[INFO] Compiling Model...")
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 1, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# Otherwise, we need to use the checkpoint model
else:
    # load the checkpoint from disk
    print("[INFO] Loading {}...".format(args["model"]))
    model = load_model(args["model"])
    # update the learning rate
    print("[INFO] Old Learning Rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] New Learning Rate: {}".format(K.get_value(model.optimizer.lr)))

# Build the path to training plot and training history
plotPath = os.path.sep.join(["output", "resnet_fashion_mnist.png"])
jsonPath = os.path.sep.join(["output", "resnet_fashion_mnist.json"])

# Construct the callback set
callbacks = [EpochCheckPoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
             TrainingMonitor(plotPath, jsonPath=jsonPath, startAt=args["start_epoch"])]

# Train the network
print("[INFO] Training the Network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // 128, epochs=80,
                    callbacks=callbacks, verbose=1)
