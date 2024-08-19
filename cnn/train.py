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
  
      # switching off autograd for evaluation
    with torch.no_grad():
  
        model.eval()
  
        for (x, y) in validDataLoader:
   
            (x, y) = (x.to(device), y.to(device))
   
            pred = model(x)
            totalValidLoss += loss(pred, y)
   
            validCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    avgTrainLoss = totalTrainLoss / trainSteps
    avgValidLoss = totalValidLoss / validSteps
    
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    validCorrect = validCorrect / len(validDataLoader.dataset)

    history["trainLoss"].append(avgTrainLoss.cpu().detach().numpy())
    history["trainAccuracy"].append(trainCorrect)
    history["validLoss"].append(avgValidLoss.cpu().detach().numpy())
    history["validAccuracy"].append(validCorrect)
    
    print("EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Valid loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValidLoss, validCorrect))
    
end = time.time()
print("time taken to train the model: {:.2f}s".format(end - start))

with torch.no_grad():
 
	model.eval()
	
	preds = []
 
	# looping over the test set
	for (x, y) in testDataLoader:
  
		x = x.to(device)
  
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
  
# generating a classification report
print(classification_report(testData.targets.cpu().numpy(), np.array(preds), target_names=testData.classes))