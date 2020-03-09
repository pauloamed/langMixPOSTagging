'''
Script for extracting infos from the POS model and datasets
Usage:
    python extractInfos.py VOCAB_PATH INFOS_PICKLE_PATH TAGS_PATH EMBEDDINGS_DIR

    VOCAB_PATH: path where vocab csv file will be saved
    INFOS_PICKLE_PATH: path to pickle file where the info object will be saved
    TAGS_PATH: path where tags csv file will be saved
    EMBEDDINGS_DIR: directory where the pickle files from the embeddings will be saved
'''

import sys
import argparse

import torch


from posembd.datasets import DatasetsPreparer, UsableDataset
from posembd.models import createPOSModel



##################################### HANDLING ARGS ###################################

parser = argparse.ArgumentParser()
parser.add_argument("-mp", "--modelPath", required = True)
parser.add_argument("-wes", "--wordEmbeddingSize", required = True)
parser.add_argument("-ces", "--charEmbeddingSize", required = True)
parser.add_argument("-bs", "--BILSTMSize", required = True)
args = parser.parse_args()


# Path to trained posembd model
modelPath = args['modelPath']

posModel = createPOSModel(args, char2id, datasets)
posModel.to(device)
