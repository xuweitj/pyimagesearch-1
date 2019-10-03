# ------------------------
#   USAGE
# ------------------------
# python extract.py --images 101_ObjectCategories --output temp_output --hashes hashes.pickle

# ------------------------
#   IMPORT
# ------------------------
from pyimagesearch.parallel_hashing import process_images
from pyimagesearch.parallel_hashing import chunk
from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils import paths
import numpy as np
import argparse
import pickle
import os

# ------------------------
#   MAIN FUNCTION
# ------------------------
# Check to see if this is the main thread of execution
if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, type=str, help="path to input directory of images")
    ap.add_argument("-o", "--output", required=True, type=str,
                    help="path to output directory to store intermediate files")
    ap.add_argument("-a", "--hashes", required=True, type=str, help="path to output hashes dictionary")
    ap.add_argument("-p", "--procs", type=int, default=-1, help="# of processes to spin up")
    args = vars(ap.parse_args())
    # Determine the number of concurrent processes to launch when distributing the load across the system
    # then create the list of process IDs.
    procs = args["procs"] if args["procs"] > 0 else cpu_count()
    proc_ids = list(range(0, procs))
    # Grab the paths to the input images, then determine the number of images each process will handle
    print("[INFO] grabbing image paths...")
    all_img_paths = sorted(list(paths.list_images(args["images"])))
    num_img_per_proc = len(all_img_paths) / float(procs)
    num_img_per_proc = int(np.ceil(num_img_per_proc))
    # Chunk the image paths into N (approximately) equal sets, one set of image paths for each individual process
    chunked_paths = list(chunk(all_img_paths, num_img_per_proc))
    # Initialize the list of payloads
    payloads = []
    # Loop over the set chunked image paths
    for (i, img_paths) in enumerate(chunked_paths):
        # Construct the path to the output intermediary file for the current process
        output_path = os.path.sep.join([args["output"], "proc_{}.pickle".format(i)])
        # Construct a dictionary of data for the payload, then add it to the payloads list
        data = {"id": i, "input_paths": img_paths, "output_path": output_path}
        payloads.append(data)
    # Construct and launch the processing pool
    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)
    # Close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")
    # Initialize the hashes dictionary (i.e., will combine the results of each pickled/serialized dictionary
    # into a single dictionary).
    print("[INFO] combining hashes...")
    hashes = {}
    # Loop over all pickle files in the output directory
    for p in paths.list_files(args["output"], validExts=".pickle",):
        # load the contents of the dictionary
        data = pickle.loads(open(p, "rb").read())
        # loop over the hashes and image paths in the dictionary
        for (tempH, tempPaths) in data.items():
            # grab all image paths with the current hash, add in the image paths for the current pickle file,
            # and then update the hash dictionary
            img_paths = hashes.get(tempH, [])
            img_paths.extend(tempPaths)
            hashes[tempH] = img_paths
    # Serialize the hashes dictionary to disk
    print("[INFO] serializing hashes...")
    f = open(args["hashes"], "wb")
    f.write(pickle.dumps(hashes))
    f.close()
