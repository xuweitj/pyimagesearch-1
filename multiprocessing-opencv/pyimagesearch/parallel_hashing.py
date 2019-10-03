# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
import numpy as np
import pickle
import cv2

# ------------------------
#   FUNCTIONS
# ------------------------
def dhash(image, hashSize=8):
    # convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# resize the input image, adding a single column (width) so we can compute the horizontal gradient
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_hash(h):
	# convert the hash to NumPy's 64-bit float and then back to Python's built in int
	return int(np.array(h, dtype="float64"))

def chunk(l, n):
	# loop over the list in n-sized chunks
	for i in range(0, len(l), n):
		# yield the current n-sized chunk to the calling function
		yield l[i: i + n]

def process_images(payload):
	# display the process ID for debugging and initialize the hashes dictionary
	print("[INFO] starting process {}".format(payload["id"]))
	hashes = {}
	# loop over the image paths
	for imagePath in payload["input_paths"]:
		# load the input image, compute the hash, and conver it
		image = cv2.imread(imagePath)
		h = dhash(image)
		h = convert_hash(h)
		# update the hashes dictionary
		l = hashes.get(h, [])
		l.append(imagePath)
		hashes[h] = l
	# serialize the hashes dictionary to disk using the supplied output path
	print("[INFO] process {} serializing hashes".format(payload["id"]))
	f = open(payload["output_path"], "wb")
	f.write(pickle.dumps(hashes))
	f.close()