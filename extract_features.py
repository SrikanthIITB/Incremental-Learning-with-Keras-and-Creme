# USAGE
# python extract_features.py --dataset train --csv features.csv

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import random
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-c", "--csv", required=True,
	help="path to output CSV file")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="batch size for the network")
args = vars(ap.parse_args())

# load the ResNet50 network and store the batch size in a convenience
# variable
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)
bs = args["batch_size"]

# grab all image paths in the input directory and randomly shuffle
# the paths
imagePaths = list(paths.list_images(args["dataset"]))
random.seed(42)
random.shuffle(imagePaths)

# extract the class labels from the image paths, then encode the
# labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# define our set of columns
cols = ["feat_{}".format(i) for i in range(0, 7 * 7 * 2048)]
cols = ["class"] + cols

# open the CSV file for writing and write the columns names to the
# file
csv = open(args["csv"], "w")
csv.write("{}\n".format(",".join(cols)))

# loop over the images in batches
for (b, i) in enumerate(range(0, len(imagePaths), bs)):
	# extract the batch of images and labels, then initialize the
	# list of actual images that will be passed through the network
	# for feature extraction
	print("[INFO] processing batch {}/{}".format(b + 1,
		int(np.ceil(len(imagePaths) / float(bs)))))
	batchPaths = imagePaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	# loop over the images and labels in the current batch
	for imagePath in batchPaths:
		# load the input image using the Keras helper utility while
		# ensuring the image is resized to 224x224 pixels
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)

		# preprocess the image by (1) expanding the dimensions and
		# (2) subtracting the mean RGB pixel intensity from the
		# ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# add the image to the batch
		batchImages.append(image)

	# pass the images through the network and use the outputs as our
	# actual features, then reshape the features into a flattened
	# volume
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)
	features = features.reshape((features.shape[0], 7 * 7 * 2048))

	# loop over the class labels and extracted features
	for (label, vec) in zip(batchLabels, features):
		# construct a row that exists of the class label and extracted
		# features
		vec = ",".join([str(v) for v in vec])
		csv.write("{},{}\n".format(label, vec))

# close the CSV file
csv.close()