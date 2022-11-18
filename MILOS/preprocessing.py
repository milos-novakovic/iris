import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms

###########

from PIL import Image

#Step 2 - Take Sample data

img = Image.open("./DATA/color_img_00.png")
H,W = img.size

#Step 3 - Convert to tensor

convert_tensor = transforms.ToTensor()

tensor_img = convert_tensor(img)

# ch 0 is Red (R)
# ch 1 is Green (G)
# ch 2 is Blue (B)



with open('./DATA/mean_and_std_of_training_set.npy', 'rb') as loading_handle:
    x_mean_bar_ch2_red = np.load(loading_handle)
    x_mean_bar_ch1_green = np.load(loading_handle)
    x_mean_bar_ch0_blue = np.load(loading_handle)
    
    x_std_bar_ch2_red = np.load(loading_handle)
    x_std_bar_ch1_green = np.load(loading_handle)
    x_std_bar_ch0_blue = np.load(loading_handle)

# RGB Format
mean_normalize = [x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue]
std_normalize = [x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue]

############




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