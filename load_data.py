import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
import zipfile
import os
import sys
from six.moves import urllib

# Global constants
ML_BASE_URL = "http://files.grouplens.org/datasets/movielens/"
DATA_BASE_PATH = "./data"

def load(dataset_type):
    dataset_path = DATA_BASE_PATH + "/" + dataset_type
    if not os.path.exists(dataset_path):
        dataset_url = ML_BASE_URL + dataset_type + ".zip"
        download_and_extract(dataset_url)
    if dataset_type == "ml-100k":
        data_path = dataset_path + "/u.data"
        raw_data = pd.read_csv(data_path, sep="\t", header=None)
        U = np.array([[u] for u in raw_data[0]])
        I = np.array([[i] for i in raw_data[1]])
        Y = list()
        for i in xrange(0,len(raw_data[2])):
            temp = [0,0,0,0,0]
            temp[raw_data[2][i]-1] = 1
            Y.append(temp)
        return U,I,np.concatenate([Y],0)

'''
Generates a batch iterator for a dataset.
'''
def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def download_and_extract(dataset_url):
  if not os.path.exists(DATA_BASE_PATH):
    os.makedirs(DATA_BASE_PATH)
  filename = dataset_url.split('/')[-1]
  filepath = os.path.join(DATA_BASE_PATH, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(dataset_url, filepath, reporthook=_progress)
    statinfo = os.stat(filepath)
    print('\nSuccessfully downloaded', filename, statinfo.st_size, 'bytes.')
    zipfile.ZipFile(filepath, 'r').extractall(DATA_BASE_PATH)

# load("ml-100k")
# load("ml-10m")