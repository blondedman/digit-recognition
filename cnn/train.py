import matplotlib
matplotlib.use("agg")

from cnn.LeNet import LeNet

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
trainData = KMNIST(root="data", train=True, download=True, transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())

# calculating the train/validation split
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValidSamples = int(len(trainData) * VALID_SPLIT)
(trainData, validData) = random_split(trainData, [numTrainSamples, numValidSamples],generator=torch.Generator().manual_seed(42))