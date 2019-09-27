# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import pickle
import cv2


# ------------------------
#   FUNCTIONS
# ------------------------
def dhash(img, hash_size=8):
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize the input image, adding a single column (width) in order to compute the horizontal gradient
    resized = cv2.resize(gray, (hash_size+1, hash_size))
    # compute the (relative) horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def convert_hash(h):
    # convert the hash to numpy 64-bit float and then back to python built in int
    return int(np.array(h, dtype="float64"))


def chunk(lst, num):
    # loop over the list in n-sized chunks
    for i in range(0, len(lst), num):
        # yield the current n-sized chunk to the calling function
        yield lst[i: i + num]


def process_images(payload):
    # display the process ID for debugging and initialize the hashes dictionary
    print("[INFO] starting process {}".format(payload["id"]))
    hashes = {}
    # loop over the image paths
    for image_path in payload["input_paths"]:
        # load the input image, compute the hash function and conver it
        img = cv2.imread(image_path)
        h = dhash(img)
        h = convert_hash(h)
        # update the hash dictionary
        lst = hashes.get(h, [])
        lst.append(image_path)
        hashes[h] = lst
    # serialize the hash dictionary to disk using the supplied output path
    print("[INFO] process serializing hashes {}".format(payload["id"]))
    f = open(payload["output_path"], "wb")
    f.write(pickle.dumps(hashes))
    f.close()