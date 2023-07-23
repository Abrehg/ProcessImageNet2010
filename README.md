\# ImageNet Data Preprocessing

This repository contains code for preprocessing the ImageNet 2010 dataset and converting it into a more manageable format using TensorFlow. The preprocessing involves loading and processing the images, saving the processed data to disk in batches, and optionally compiling the batches into a single file.

\## Requirements

Before running the code, ensure you have the following dependencies installed:

\- TensorFlow

\- h5py

\- numpy

\## Usage

1\. Import the required libraries:

\```python

import tensorflow as tf

import os

import h5py

import numpy as np

import gc

import multiprocessing

from inputFormatting import formatImg

from inputFormatting import formatTensorFromPath

\```

2\. Define the main function that preprocesses the dataset:

\```python

def main(image\_folder, dump\_folder):

`    `# Load and process train set, save to disk in batches

`    `load\_and\_process\_images(image\_folder, 'train', dump\_folder, batch\_size=1024)

`    `# Load and process validation set, save to disk in batches

`    `load\_and\_process\_images(image\_folder, 'val', dump\_folder, batch\_size=1024)

`    `# Load and process test set, save to disk in batches

`    `#load\_and\_process\_images(image\_folder, 'test', dump\_folder, batch\_size=1024)

\```

3\. Define the function to load and process images and save the data to disk in batches:

\```python

def load\_and\_process\_images(dir\_path, data\_folder, pickleFolder, batch\_size):

`    `# Use multiprocessing to speed up the processing

`    `pool = multiprocessing.Pool(processes=multiprocessing.cpu\_count())

`    `image\_files = []

`    `batch\_idx = 0

`    `image\_num = 0

`    `image\_folder = os.path.join(dir\_path, data\_folder)

`    `# Iterate through each class folder

`    `for folder\_name in os.listdir(image\_folder):

`        `class\_folder = os.path.join(image\_folder, folder\_name)

`        `if os.path.isdir(class\_folder):

`            `for file\_name in os.listdir(class\_folder):

`                `image\_num = image\_num + 1

`                `image\_path = os.path.join(class\_folder, file\_name)

`                `image\_files.append(image\_path)

`                `if image\_num % batch\_size == 0:

`                    `batch\_data = pool.map(process\_image, image\_files)

`                    `embeddings, labels = zip(\*batch\_data)

`                    `save\_data\_to\_disk(embeddings, labels, data\_folder, batch\_idx, pickleFolder)

`                    `image\_files = []

`                    `batch\_idx = batch\_idx + 1

`                    `batch\_data = None

`                    `embeddings = None

`                    `labels = None

`                    `gc.collect()

`    `if len(image\_files) != 0:

`        `batch\_data = pool.map(process\_image, image\_files)

`        `embeddings, labels = zip(\*batch\_data)

`        `save\_data\_to\_disk(embeddings, labels, data\_folder, batch\_idx, pickleFolder)

`        `batch\_data = None

`        `embeddings = None

`        `labels = None

`        `gc.collect()

\```

4\. Define the function to process individual images and extract embeddings and labels:

\```python

def process\_image(image\_path):

`    `# Extract relevant information from the image path

`    `path = image\_path.split("/")

`    `for i in range(0, len(path)):

`        `if path[i] == "ImageNet":

`            `imgFolder = image\_path.split("/")[(i+1)]

`            `mainFolder = image\_path.split("/")[:(i+1)]

`            `break

`    `mainFolderPath = ""

`    `for j in range(0, len(mainFolder)):

`        `mainFolderPath = mainFolderPath + mainFolder[j]

`    `data\_folder = os.path.join(mainFolderPath, 'data')

`    `# Process images based on their folder (train, val, or test)

`    `if imgFolder == 'train':

`        `finalEmbeddings, other = formatImg(formatTensorFromPath(image\_path))

`        `raw\_label = os.path.basename(image\_path).split('.')[0][1:]

`        `i = raw\_label.find("\_")

`        `class\_label = int(raw\_label[(i+1):])

`        `one\_hot\_label = tf.one\_hot(class\_label, depth=1000)

`    `elif imgFolder == 'val':

`        `labelPath = os.path.join(data\_folder, 'ILSVRC2010\_validation\_ground\_truth.txt')

`        `finalEmbeddings, other = formatImg(formatTensorFromPath(image\_path))

`        `imgIndex = int(os.path.basename(path).split('.')[0].split("\_")[-1])

`        `label\_list = read\_file\_as\_list(labelPath)

`        `class\_label = label\_list[imgIndex]

`        `one\_hot\_label = tf.one\_hot(class\_label, depth=1000)

`    `elif imgFolder == 'test':

`        `labelPath = os.path.join(data\_folder, 'ILSVRC2010\_testing\_ground\_truth.txt')

`        `finalEmbeddings, other = formatImg(formatTensorFromPath(image\_path))

`        `imgIndex = int(os.path.basename(path).split('.')[0].split("\_")[-1])

`        `label\_list = read\_file\_as\_list(labelPath)

`        `class\_label = label\_list[imgIndex]

`        `one\_hot\_label = tf.one\_hot(class\_label, depth=1000)

`    `return finalEmbeddings, one\_hot\_label

\```

5\. Define the function to save processed data to disk in HDF5 format:

\```python

def save\_data\_to\_disk(embeddings, labels, dataset, batchNum, folder):

`    `filename = f"{dataset}\_batch{batchNum}.h5"

`    `hdf5\_file\_path = os.path.join(folder, filename)

`    `with h5py.File(hdf5\_file\_path, "w") as file:

`        `# Save each embedding and label separately for each image

`        `for idx, (emb, lab) in enumerate(zip(embeddings, labels)):

`            `emb\_name = f"embedding\_{idx}"

`            `lab\_name = f"label\_{idx}"

`            `file.create\_dataset(emb\_name, data=emb.numpy())

`            `file.create\_dataset(lab\_name, data=lab.numpy())

`    `print(f"{filename} has been saved")

\```

6\. Optionally, you can compile the batches into a single pickle file using the following functions:

\```python

def compile\_data\_from\_batches(data\_folder, dataset\_name):

`    `compiled\_data = []

`    `batch\_idx = 0

`    `while True:

`        `batch\_filename = os.path.join(data\_folder, f"{dataset\_name}\_batch{batch\_idx}.h5")

`        `if os.path.exists(batch\_filename):

`            `with h5py.File(batch\_filename, "r") as file:

`                `batch\_data = list(zip(file["embeddings"], file["labels"]))

`            `compiled\_data.extend(batch\_data)

`            `# possibly remove

`            `os.remove(batch\_filename)

`            `batch\_idx += 1

`        `else:

`            `break

`    `compiled\_filename = os.path.join(data\_folder, f"{dataset\_name}\_compiled.h5")

`    `with h5py.File(compiled\_filename, "w") as file:

`        `embeddings = file.create\_dataset("embeddings", data=[item[0] for item in compiled\_data])

`        `labels = file.create\_dataset("labels", data=[item[1] for item in compiled\_data])

\```

\## Notes

\- This code preprocesses the ImageNet 2010 dataset into batches of embeddings and labels, saving them to disk in the specified dump folder. The data is saved in HDF5 format.

\- The preprocessing for the test set is commented out by default since it is highly RAM intensive. Uncomment it only if your machine can handle it.

\- The optional compilation of data from batches into a single file can be performed by uncommenting the respective lines in the `main` function.

\- Make sure to provide

` `the correct paths to the `image\_folder` and `dump\_folder` when calling the `main` function.

Please feel free to contact the author of this code for any questions or issues related to the preprocessing process. Happy coding!

