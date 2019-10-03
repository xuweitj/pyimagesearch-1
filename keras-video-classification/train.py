# ------------------------
#   USAGE
# ------------------------
# python train.py --dataset Sports-Type-Classifier/data --model model/activity.model \
# --label-bin model/lb.pickle --epochs 50

# ------------------------
#   IMPORT
# ------------------------
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25, help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Initialize the set of labels from the spots activity dataset we are going to train our network on
LABELS = {"weight_lifting", "tennis", "football"}

# Grab the list of images in our dataset directory, then initialize the list of data (i.e, images) and class images
print("[INFO] Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # if the label of the current image is not part of the labels are interest in, then ignore the image
    if label not in LABELS:
        continue
    # load the image, convert it to RGB channel ordering, and resize it to be a fixed 224x224 pixels,
    # ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    # update the data and labels list respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)
# perform on-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 75% of the data for training and 25% of the data for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True,fill_mode="nearest"
)
# initialize the validation/testing data augmentation object (which we'll be adding mean subtraction to)
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the mean subtraction value for each of the data
# augmentation objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean
# load the ResNet-50 network, ensuring the head FC layer sets are left off
baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will not be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False
# compile our model (this needs to be done after our setting our layers to being non-trainable)
print("[INFO] Compiling Model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers
# to start to become initialized with actual "learned" values versus pure random
print("[INFO] Training Head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=32), steps_per_epoch=len(trainX) // 32,
    validation_data=valAug.flow(testX, testY), validation_steps=len(testX) // 32,
    epochs=args["epochs"])
# evaluate the network
print("[INFO] Evaluating Network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
# serialize the model to disk
print("[INFO] Serializing Network...")
model.save(args["model"])
# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
