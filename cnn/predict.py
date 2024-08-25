import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader
from torch.utils.data import Subset as sub
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testData = KMNIST(root="cnn", train=False, download=True, transform=ToTensor())

idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = sub(testData, idxs)

testDataLoader = DataLoader(testData, batch_size=1)

model = torch.load("cnn/model.pth").to(device)
model.eval()

# switching off autograd
with torch.no_grad():
    
    # looping over test set:
    for (image, label) in testDataLoader:
        
        orig = image.numpy().squeeze(axis=(0, 1))
        gtlabel = testData.dataset.classes[label.numpy()[0]]
        
        image = image.to(device)
        pred = model(image)
        
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        predlabel = testData.dataset.classes[idx]
        
        orig = np.dstack([orig] * 3)
        orig = imutils.resize(orig, width=128)
        
        color = (0, 255, 0) if gtlabel == predlabel else (0, 0, 255)
        cv2.putText(orig, gtlabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        
        print("ground truth label: {}, predicted label: {}".format(gtlabel, pred))
        cv2.imshow("image", orig)
        cv2.waitKey(0)