# ------------------------
#   USAGE
# ------------------------
# python download-images.py --urls urls.txt --output images/santa

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

# Grab the list of URLs from the input file, then initialize the total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

# Loop the URLs
for url in rows:
    try:
        # try to download the image
        r = requests.get(url, timeout=60)
        # save the image to disk
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(8))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()
        # update the counter
        print("[INFO] Downloaded: {}".format(p))
        total += 1
    # handle if any exceptions are thrown during the download process
    except:
        print("[INFO] Error downloading {}...skipping".format(p))

# Loop over the downloaded image paths
for imgPath in paths.list_images(args["output"]):
    # initialize if the image should be deleted or not
    delete = False
    # try to load the images
    try:
        img = cv2.imread(imgPath)
        # if the image is 'None' then we could not properly load it from disk so delete it
        if img is None:
            print("None")
            delete = True

    # if OpenCV cannot load the image then the image is likely corrupt and so we should delete it
    except:
        print("Except")
        delete = True
    # check to see if image should be deleted
    if delete:
        print("[INFO] Deleting {}".format(imgPath))
        os.remove(imgPath)

