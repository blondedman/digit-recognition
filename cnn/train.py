import matplotlib
matplotlib.use("agg")

import LeNet

from sklearn.metrics import classification_report

from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

TRAIN_SPLIT = 0.75
VALID_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading  the KMNIST dataset
trainData = KMNIST(root="cnn", train=True, download=True, transform=ToTensor())
testData = KMNIST(root="cnn", train=False, download=True, transform=ToTensor())

# calculating the train/validation split
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValidSamples = int(len(trainData) * VALID_SPLIT)
(trainData, validData) = random_split(trainData, [numTrainSamples, numValidSamples],generator=torch.Generator().manual_seed(42))

# initializing the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
validDataLoader = DataLoader(validData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculating steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
validSteps = len(validDataLoader.dataset) // BATCH_SIZE

# initializing the LeNet model
model = LeNet(channels=1, classes=len(trainData.dataset.classes)).to(device)

# initializing the optimizer anf loss function
optimizer = Adam(model.parameters(), lr = INIT_LR)
loss = nn.NLLLoss()

# initializing a dictionary to store training historg
history = {"trainLoss": [], "trainAccuracy": [], "validLoss": [], "validAccuracy": []}

# measuring training time
start = time.time()

# looping over our epochs
for e in range(0, EPOCHS):
    
	model.train()
 
	totalTrainLoss = 0
	totalValidLoss = 0
 
	trainCorrect = 0
	validCorrect = 0
 
	# looping over the training set
	for (x, y) in trainDataLoader:
     
		(x, y) = (x.to(device), y.to(device))
  
		pred = model(x)
		loss = loss(pred, y)
  
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
  
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()