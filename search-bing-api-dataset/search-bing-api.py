# ------------------------
#   USAGE
# ------------------------
# python search-bing-api.py --query "charmander" --output dataset/charmander
# python search-bing-api.py --query "pikachu" --output dataset/pikachu
# python search-bing-api.py --query "squirtle" --output dataset/squirtle
# python search-bing-api.py --query "bulbasaur" --output dataset/bulbasaur
# python search-bing-api.py --query "mewtwo" --output dataset/mewtwo

# ------------------------
#   IMPORTS
# ------------------------
# import the necessary packages
from requests import exceptions
import argparse
import requests
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

# Set your Microsoft Cognitive Services API key along with:
# (1) The maximum number of results for a given search;
# (2) The group size for results (maximum of 50 per request)
API_KEY = "400afc25e7aa4fa28be594b7ae3e7f5c"
MAX_RESULTS = 250
GROUP_SIZE = 50

# Set the endpoint API Url
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# When attempting to download images from the web both the Python programming language and the request library have a
# number of exceptions that can be thrown so let's build a list of them now so we can filter on them
EXCEPTIONS = {IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError,
              exceptions.Timeout}

# Store the search term in a convenience variable then set the headers and search parameters
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# Make the search
print("[INFO] Searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# Grab the results from the search, including the total number of estimated results returned by the Bing API
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults, term))

# Initialize the total number of images downloaded thus far
total = 0

# Loop over the estimated number of results in 'GROUP_SIZE' groups
for offset in range(0, estNumResults, GROUP_SIZE):
    # update the search parameters using the current offset, then make the request to fetch the results
    print("[INFO] Making request for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] Saving images for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, estNumResults))
    # loop over the results
    for v in results["value"]:
        # try to download the image
        try:
            # make the request to download the image
            print("[INFO] Fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)
            # build the path to the ouput image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])
            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()
        # catch any errors that would not unable us to download the image
        except Exception as e:
            # check to see if the exception is in our list of exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] Skipping: {}".format(v["contentUrl"]))
                continue
        # try to load the image from disk
        img = cv2.imread(p)
        # if the image is 'None' then we could not properly load the image from disk so it is better to delete it
        if img is None:
            print("[INFO] Deleting: {}".format(p))
            os.remove(p)
            continue
        # update the counter
        total += 1
