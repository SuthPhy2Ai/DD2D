import re
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import minimize
from torch.utils.data import Dataset
from torch.nn import functional as F
from numpy import * # to override the math functions
from matplotlib import pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out



@torch.no_grad()
def pred_from_model(model, x, pattern=None, itos=None, topN=3):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    x = x.to('cpu')
    pattern = pattern.to('cpu')
    model = model.to('cpu')
    model.eval()
    _, similarity = model(x, pattern=pattern)
    similarity = similarity.numpy()
    topN = -1 * (topN + 1)
    top3_indices = np.argsort(similarity, axis=1)[:, :topN:-1]
    np.savetxt('topN_index.txt', top3_indices)
    N = shape(similarity)[0]
    true_captions = np.arange(N)
    accuracy_sum = 0
    error = []
    misunderstand = []
    for i in range(N):
        top3_captions = [true_captions[j] for j in top3_indices[i]]
        if true_captions[i] in top3_captions:
            accuracy_sum += 1
        else:
            if topN == -6:
              err = x[i,:].numpy()
              misunderstood = top3_indices[i, 0]
              misunder = x[misunderstood, :].numpy()
              misunderstand.append(misunder)
              error.append(err)
    accuracy = accuracy_sum / N * 100
    return accuracy



@torch.no_grad()
def sample_from_model(model, x, pattern=None, itos=None, topN=3):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    x = x.to('cpu')
    pattern = pattern.to('cpu')
    model = model.to('cpu')
    model.eval()
    # outputs = model(x, pattern=pattern)
    # print(f"Model outputs: {outputs.shape}")
    _, similarity = model(x, pattern=pattern)
    similarity = similarity.numpy()
    topN = -1 * (topN + 1)
    top3_indices = np.argsort(similarity, axis=1)[:, :topN:-1]
    N = shape(similarity)[0]
    true_captions = np.arange(N)
    accuracy_sum = 0
    error = []
    misunderstand = []
    for i in range(N):
        top3_captions = [true_captions[j] for j in top3_indices[i]]
        if true_captions[i] in top3_captions:
            accuracy_sum += 1
        else:
            if topN == -6:
              err = x[i,:].numpy()
              misunderstood = top3_indices[i, 0]
              misunder = x[misunderstood, :].numpy()
              misunderstand.append(misunder)
              error.append(err)
    accuracy = accuracy_sum / N * 100
    return accuracy



class CharDataset(Dataset):
    def __init__(self, data, block_size,
                 numVars, numYs, numPoints):

        data_size = len(data)
        print('data has %d examples' % (data_size))

        self.numVars = numVars
        self.numYs = numYs
        self.numPoints = numPoints

        

        
        self.block_size = block_size
        self.data = data # it should be a list of examples

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab an example from the data
        chunk = self.data[idx] # sequence of tokens including x, y, eq, etc.
        
        try:
            chunk = json.loads(chunk) # convert the sequence tokens to a dictionary
        except Exception as e:
            print("Couldn't convert to json: {} \n error is: {}".format(chunk, e))
            # try the previous example
            idx = idx - 1 
            idx = idx if idx>=0 else 0
            chunk = self.data[idx]
            chunk = json.loads(chunk) # convert the sequence tokens to a dictionary
            
        # find the number of variables in the equation
        printInfoCondition = random.random() < 0.0000001
        points = torch.zeros(self.numVars + self.numYs, self.numPoints)
        xyz = chunk['X']
        x = np.array(xyz[0])
        y = np.array(xyz[1])
        z = np.array(xyz[2])
        length = len(x)
        ################## padding=0.0 cls=3.0 ###############
        points[0, :length] = torch.tensor(x)
        points[1, :length] = torch.tensor(y)
        points[2, :length] = torch.tensor(z)
        points[2, length] = 3.0
        ########### padding=-1 cls=200.0 ####################
        seq = chunk['seq']
        seq = [x for x in seq if x != '']
        seq = np.array(seq, dtype='float32')

        # seq[0], seq[3] = seq[3], seq[0]
        # seq[1], seq[4] = seq[4], seq[1]
        # seq[2], seq[5] = seq[5], seq[2]

        seq_back = -1.0 * torch.ones(self.block_size)

        # seq_back[0:len(seq)] = torch.tensor(seq)
        # seq_back[len(seq)] = 210.0

        ################ 额外cls ###############
        seq_back[1:len(seq)+1] = torch.tensor(seq)
        seq_back[len(seq)+1] = 210.0
        seq_back[0] = 200.0



        return points, seq_back

def processDataFiles(files):
    text = ""
    for f in tqdm(files):
        with open(f, 'r') as h: 
            lines = h.read() # don't worry we won't run out of file handles
            if lines[-1]==-1:
                lines = lines[:-1]
            #text += lines #json.loads(line)    
            text = ''.join([lines,text])    
    return text
