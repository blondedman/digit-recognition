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