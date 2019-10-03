# ------------------------
#   USAGE
# ------------------------
# python index_images.py --images 101_ObjectCategories --tree vptree.pickle --hashes hashes.pickle

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
from pyimagesearch.hashing import convert_hash
from pyimagesearch.hashing import hamming
from pyimagesearch.hashing import dhash
from imutils import paths
import argparse
import pickle
import vptree
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, type=str, help="path to input directory of images")
ap.add_argument("-t", "--tree", required=True, type=str, help="path to output VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str, help="path to output hashes dictionary")
args = vars(ap.parse_args())

# Grab the paths to the input images and initialize the dictionary hashes
imagePaths = list(paths.list_images(args["images"]))
hashes = {}

# Loop over the image paths
for (i, imgPath) in enumerate(imagePaths):
    # Load the input image
    print("[INFO] Processing image {}/{}".format(i+1, len(imagePaths)))
    img = cv2.imread(imgPath)
    # Compute the hash for the image and convert it
    h = dhash(img)
    h = convert_hash(h)
    # Update the hashes dictionary
    l = hashes.get(h, [])
    l.append(imgPath)
    hashes[h] = l

# Build the VP-Tree
print("[INFO] Building VP-Tree...")
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming)

# Serialize the VP-Tree to disk
print("[INFO] Serializing VP-Tree...")
f = open(args["tree"], "wb")
f.write(pickle.dumps(tree))
f.close()

# Serialize the hashes to dictionary
print("[INFO] Serializing hashes...")
f = open(args["hashes"], "wb")
f.write(pickle.dumps(hashes))
f.close()
