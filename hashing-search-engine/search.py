# ------------------------
#   USAGE
# ------------------------
# python search.py --tree vptree.pickle --hashes hashes.pickle --query image_queries/accordion.jpg

# ------------------------
#   IMPORT
# ------------------------
# import the necessary packages
from pyimagesearch.hashing import convert_hash
from pyimagesearch.hashing import dhash
import argparse
import pickle
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tree", required=True, type=str, help="path to pre-constructed VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str, help="path to hashes dictionary")
ap.add_argument("-q", "--query", required=True, type=str, help="path to input query image")
ap.add_argument("-d", "--distance", type=int, default=10, help="maximum hamming distance")
args = vars(ap.parse_args())

# Load the VP-Tree and hashes the dictionary
print("[INFO] Loading VP-Tree and hashes...")
tree = pickle.loads(open(args["tree"], "rb").read())
hashes = pickle.loads(open(args["hashes"], "rb").read())

# Load the input query images
img = cv2.imread(args["query"])
cv2.imshow("Query", img)

# Compute the hash for the query image, then convert it
queryHash = dhash(img)
queryHash = convert_hash(queryHash)

# Perform the search
print("[INFO] Performing search...")
start = time.time()
results = tree.get_all_in_range(queryHash, args["distance"])
results = sorted(results)
end = time.time()
print("[INFO] Search took {} seconds".format(end - start))

# Loop over the results
for (d, h) in results:
    # grab all image paths in our dataset with the same hash
    resultPaths = hashes.get(h, [])
    print("[INFO] {} total image(s) with d: {}, h: {}".format(len(resultPaths), d, h))
    # loop over the result paths
    for resultPath in resultPaths:
        # load the result image and display it to our screen
        result = cv2.imread(resultPath)
        cv2.imshow("Result", result)
        cv2.waitKey(0)