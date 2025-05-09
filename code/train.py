#!/usr/bin/env python
# coding: utf-8

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# load libraries
import os
import csv
import glob
import json
import math
import pickle
import random
import numpy as np
#from tqdm import tqdm
from numpy import * # to override the math functions
import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import Dataset

from utils import set_seed, sample_from_model
from trainer import Trainer, TrainerConfig
from models_new import DD2D, TransformerConfig
from utils import processDataFiles, CharDataset

# set the random seed
set_seed(42)

# config
device='gpu'
scratch=True 
numEpochs = 2000 
embeddingSize = 384 
numPoints = [20,350] 
numVars = 2 
numYs = 1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 180 # spatial extent of the model for its context
batchSize = 256 # batch size of training data
target = 'Skeleton' #'Skeleton' #'EQ'
dataDir = './datasets/'

data_set = 0
if data_set == 0:
    dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize'.format(numVars, numPoints[0], numPoints[1], embeddingSize)
else:
    dataInfo = 'XYE_{}Var_{}-{}Points_{}EmbeddingSize_dataset{}'.format(numVars, numPoints[0], numPoints[1], embeddingSize, data_set)

titleTemplate = "{} equations of {} variables - Benchmark"
addr = './SavedModels/' # where to save model
n_layer = 3
n_head = 4
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
addVars = True if variableEmbedding == 'STR_VAR' else False
maxNumFiles = 100 # maximum number of file to load in memory for training the neural network
bestLoss = None # if there is any model to load as pre-trained one
fName = '{}_DD2D_{}_{}_{}_MINIMIZE.txt'.format(dataInfo, 
                                             'layer_heads_{}_{}'.format(n_layer, n_head), 
                                             'Padding',
                                             variableEmbedding)
ckptPath = '{}/{}.pt'.format(addr,fName.split('.txt')[0])
try: 
    os.mkdir(addr)
except:
    print('Folder already exists!')

# load the train dataset
train_file = 'train_dataset_{}.pb'.format(fName)
if os.path.isfile(train_file) and not scratch:
    # just load the train set
    with open(train_file, 'rb') as f:
        train_dataset,trainText,chars = pickle.load(f)
else:
    # process training files from scratch

    ##########pre-processing .dat##################
    # path = '../imgs_train/'
    # files = glob.glob(os.path.join(path, '*.dat'))
    # Points = []
    # fileID = 0
    # template = {'X': []}
    # for idx, file_path in enumerate(files):
    #     structure = template.copy()
    #     points = np.load(file_path, allow_pickle=True)
    #     indx = np.where(points != 0.0)
    #     values = points[indx]
    #     result = np.zeros((3, len(indx[0])))
    #     result[0, :] = indx[0]
    #     result[1, :] = indx[1]
    #     result[2, :] = np.log(values)
    #     with open('../aimd_train.csv', 'r', newline='') as csvfile:
    #         reader = csv.reader(csvfile)
    #         for id_in, row in enumerate(reader):
    #             if id_in == idx:
    #                 structure['seq'] = row
    #                 break
    #     structure['X'] = result.tolist()
    #     outputpath = './dataset/{}.json'.format(fileID)
    #     if os.path.exists(outputpath):
    #         fileSize = os.path.getsize(outputpath)
    #         if fileSize > 500000000:  # 500 MB
    #             fileID += 1
    #     with open(outputpath, "a", encoding="utf-8") as h:
    #         json.dump(structure, h, ensure_ascii=False)
    #         h.write('\n')
    #     Points.append(points)
    ##########pre-processing .csv##################


    ############read json######################
    path = f'./dataset/{data_set}.json'
    files = glob.glob(path)[:maxNumFiles]
    text = processDataFiles(files)

    # chars = sorted(list(set(text)) + ['_', 'T', '<', '>', ':'])
    text = text.split('\n') # convert the raw text to a set of examples
    trainText = text[:-1] if len(text[-1]) == 0 else text

    random.shuffle(trainText) # shuffle the dataset, it's important specailly for the combined number of variables experiment
    length = []
    length_seq = []
    values = []
    for text in trainText:
        text = json.loads(text)
        text_x = text['X'][2]
        len_i = len(text_x)
        text_seq = text['seq']
        text_seq = [x for x in text_seq if x != '']
        len_j = len(text_seq)
        length.append(len_i)
        length_seq.append(len_j)
        if len_i != 0:
            value = np.max(text)
            values.append(value)
    numPoints = np.max(length) + 3
    
    val_size = int(len(trainText) * 0.1)
    val_set = trainText[:val_size]
    test_set = trainText[val_size : 2 * val_size]
    # test_set = trainText[:500]
    train_set = trainText[2 * val_size:]
    train_dataset = CharDataset(train_set, blockSize, numVars=numVars,
                    numYs=numYs, numPoints=numPoints)
    val_dataset = CharDataset(val_set, blockSize, numVars=numVars,
                    numYs=numYs, numPoints=numPoints)
    test_dataset = CharDataset(test_set, blockSize, numVars=numVars,
                    numYs=numYs, numPoints=numPoints)
    # with open(train_file, 'wb') as f:
    #     pickle.dump([train_dataset,trainText,chars], f)

lens_seq = []
# print a random sample
# for idx in range(train_dataset.__len__()):
#     points, seq = train_dataset.__getitem__(idx)
#     len_seq = len(seq)
#     lens_seq.append(len_seq)

idx = np.random.randint(train_dataset.__len__())
points, seq = train_dataset.__getitem__(idx)
print('seq:{}'.format(seq))



# create the model
mconf = TransformerConfig(train_dataset.block_size,
                  n_layer=n_layer, n_head=n_head, n_embd=embeddingSize)
model = DD2D(mconf)
    
# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=numEpochs, batch_size=batchSize, 
                      learning_rate=4e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2*len(train_dataset)*blockSize,
                      num_workers=0, ckpt_path=ckptPath)
trainer = Trainer(model, train_dataset, val_dataset, tconf, bestLoss, device=device)

# # load the best model before training
# print('The following model {} has been loaded!'.format(ckptPath))
# model.load_state_dict(torch.load(ckptPath))
# model = model.eval().to(trainer.device)

try:
    trainer.train()
except KeyboardInterrupt:
    print('KeyboardInterrupt')

# load the best model
print('The following model {} has been loaded!'.format(ckptPath))
checkpoint = torch.load(ckptPath)
model.load_state_dict(checkpoint)
model = model.eval().to(trainer.device)


loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    pin_memory=True,
    batch_size=len(test_dataset),
    num_workers=0)

for i, (x, y) in enumerate(loader):
    topN = 1
    acc = sample_from_model(model, y, x, topN=topN)
    print('Top {} accuracy: {}'.format(topN, acc))

for i, (x, y) in enumerate(loader):
    topN = 3
    acc = sample_from_model(model, y, x, topN=topN)
    print('Top {} accuracy: {}'.format(topN, acc))
for i, (x, y) in enumerate(loader):
    topN = 5
    acc = sample_from_model(model, y, x, topN=topN)
    print('Top {} accuracy: {}'.format(topN, acc))

for i, (x, y) in enumerate(loader):
    topN = 10
    acc = sample_from_model(model, y, x, topN=topN)
    print('Top {} accuracy: {}'.format(topN, acc))
