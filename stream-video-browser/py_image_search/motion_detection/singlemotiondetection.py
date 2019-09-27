# Import the necessary packages

# ------------------------
#   IMPORTS
# ------------------------
import numpy as np
import imutils
import cv2


# ------------------------
#   SingleMotionDetector
# ------------------------
class SingleMotionDetector:
    def __init__(self, accum_weight=0.5):
        # store the accumulated weight factor
        self.accum_weight = accum_weight
        # initialize the background model
        self.background = None

    def update(self, image):
        # if the background model is None, initialize it
        if self.background is None:
            self.background = image.copy().astype("float")
            return
        # update the background model by accumulating the weighted average
        cv2.accumulateWeighted(image, self.background, self.accum_weight)

    def detect(self, image, tval=25):
        # compute the absolute difference between the background model
        # and the image passed in, then threshold the delta image
        delta = cv2.absdiff(self.background.astype("uint8"), image)
        threshold = cv2.threshold(delta, tval, 255, cv2.THRESH_BINARY)[1]
        # perform a series of erosions and dilation to remove small blobs
        threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)
        # find contours in the thresholded image and initialize the
        # minimum and maximum bounding box regions for motion
        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        # if there are no contours found, return None
        if len(contours) == 0:
            return None
        # otherwise, loop over the contours
        for c in contours:
            # compute the bounding box of the contour and use it to update the
            # minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x+w), max(maxY, y+h))
        # otherwise, return a tuple of the thresholded image along with the bounding box
        return threshold, (minX, minY, maxX, maxY)



