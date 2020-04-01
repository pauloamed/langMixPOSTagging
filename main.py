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

import posembd
from posembd.models import createPOSModel

from posembd_utils import *
from utils import *
from globals import DATASETS as datasets
from globals import DATASETS_DIR as datasetsDir

from datetime import datetime


torch.set_printoptions(threshold=10000)

# Seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################### HANDLING ARGS ###################################

DEFAULT_EPOCHS = 55
DEFAULT_BATCH_SIZE = 32
DEFAULT_GRAD_CLIPPING = 25


parser = argparse.ArgumentParser()
parser.add_argument("-mp", "--modelPath", required = True)
parser.add_argument("-wes", "--wordEmbeddingSize", required = True)
parser.add_argument("-ces", "--charEmbeddingSize", required = True)
parser.add_argument("-pes", "--posEmbeddingSize", required = True)
parser.add_argument("-e", "--epochs", required = False)
parser.add_argument("-bs", "--batchSize", required = False)
parser.add_argument("-gc", "--gradClipping", required = False)
parser.add_argument("-t", "--testOnly", action = "store_true")
args = parser.parse_args()

parameters = {
    'epochs': int(args.epochs) if args.epochs is not None else DEFAULT_EPOCHS,
    'batchSize': int(args.batchSize) if args.batchSize is not None else DEFAULT_BATCH_SIZE,
    'gradClipping': int(args.gradClipping) if args.gradClipping is not None else DEFAULT_GRAD_CLIPPING,
}

# Path to trained posembd model
modelPath = args.modelPath
charEmbeddingSize = int(args.charEmbeddingSize)
wordEmbeddingSize = int(args.wordEmbeddingSize)
posEmbeddingSize = int(args.posEmbeddingSize)

testOnly = args.testOnly


datatimeNow = datetime.now()
log = "Starting training script at {}\n".format(datatimeNow) + \
        "Model will be saved at {}\n".format(modelPath) + \
        "WES: {}, CES: {}, PES: {}\n".format(wordEmbeddingSize, charEmbeddingSize, posEmbeddingSize) + \
        "Epochs: {}, batchSize: {}, gradClipping: {}".format(parameters['epochs'], parameters['batchSize'], parameters['gradClipping'])

print(log)
printToFile(log, 'log_{}.txt'.format(datatimeNow))

datasets, id2char, char2id = loadDatasets(datasets, datasetsDir)

for d in datasets:
    print(d)

posModel = createPOSModel(charEmbeddingSize, wordEmbeddingSize, posEmbeddingSize, char2id, datasets)
posModel.to(device)


if not testOnly:
    train(device, posModel, modelPath, datasets, parameters)

posModel.load_state_dict(torch.load(args.modelPath, map_location=device))

accuracy(device, posModel, datasets)
