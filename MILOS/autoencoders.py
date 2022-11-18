import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms

# Hyper parameters
num_epochs = 20
batchsize = 100
lr = 0.001

EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.003

TRAIN_DATA_PATH = './DATA/'
TEST_DATA_PATH = './DATA_TEST/'
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=TRAIN_DATA_PATH,
                                           batch_size=batchsize,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=TEST_DATA_PATH,
                                          batch_size=batchsize,
                                          shuffle=False)

