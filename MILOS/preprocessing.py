import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import cv2
import numpy as np 
import os
###########

from PIL import Image

def find_mean_std(TOTAL_NUMBER_OF_IMAGES, image_ids_numbers = None):
    # image_ids_numbers is the list of numbers for what images to produce estimate of mean and variance
    # (so that it works for full batch, i.e. all training images, when image_ids_numbers = None)
    # and for a subset of images when image_ids_numbers has acctually elements
    if image_ids_numbers == None:
        # full batch
        image_ids = [str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES))) for image_id in range(TOTAL_NUMBER_OF_IMAGES)]
    else:
        # mini-batch
        image_ids = [str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES))) for image_id in image_ids_numbers]

    #empirical calculation of first centered moment per channel of image
    x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue = \
    0.,0.,0.#np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT)

    #empirical calculation of second centered moment per channel of image
    x2_mean_bar_ch2_red, x2_mean_bar_ch1_green, x2_mean_bar_ch0_blue = \
    0.,0.,0.#np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT)

    for i,image_id in enumerate(image_ids):
        # update empirical estimates for 1st and 2nd centered moments
        image_full_path = './DATA/color_img_' + image_id + '.png'
        
        x = cv2.imread(image_full_path)

        x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue = \
            x_mean_bar_ch2_red   + (-x_mean_bar_ch2_red   + np.mean(np.float64(x)[:,:,2]))/(i+1), \
            x_mean_bar_ch1_green + (-x_mean_bar_ch1_green + np.mean(np.float64(x)[:,:,1]))/(i+1), \
            x_mean_bar_ch0_blue  + (-x_mean_bar_ch0_blue  + np.mean(np.float64(x)[:,:,0]))/(i+1)
        
        x2_mean_bar_ch2_red, x2_mean_bar_ch1_green, x2_mean_bar_ch0_blue = \
            x2_mean_bar_ch2_red     + (-x2_mean_bar_ch2_red   + np.mean(np.float64(x)[:,:,2]**2))/(i+1), \
            x2_mean_bar_ch1_green   + (-x2_mean_bar_ch1_green + np.mean(np.float64(x)[:,:,1]**2))/(i+1), \
            x2_mean_bar_ch0_blue    + (-x2_mean_bar_ch0_blue  + np.mean(np.float64(x)[:,:,0]**2))/(i+1)

    #available
    #x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue

    x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue = \
        np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch2_red   - x_mean_bar_ch2_red**2)),  \
        np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch1_green - x_mean_bar_ch1_green**2)),\
        np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch0_blue  - x_mean_bar_ch0_blue**2))  

    # with open('./DATA/mean_and_std_of_training_set.npy', 'wb') as saving_handle:
    #     np.save(saving_handle, x_mean_bar_ch2_red)
    #     np.save(saving_handle, x_mean_bar_ch1_green)
    #     np.save(saving_handle, x_mean_bar_ch0_blue)
    #     np.save(saving_handle, x_std_bar_ch2_red)
    #     np.save(saving_handle, x_std_bar_ch1_green)
    #     np.save(saving_handle, x_std_bar_ch0_blue)

    # two arrays of (3,) size
    RGB_mean = [x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue]#np.dstack((x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue))
    RGB_std = [x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue]#np.dstack((x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue))    
    return RGB_mean, RGB_std


TEST_DATA_PATH = './DATA_TEST/'
#Step 2 - Take Sample data

img = Image.open("./DATA/color_img_000.png")
H,W = img.size

#Step 3 - Convert to tensor

#convert_tensor = transforms.ToTensor()

#tensor_img = convert_tensor(img)

# ch 0 is Red (R)
# ch 1 is Green (G)
# ch 2 is Blue (B)
TRAIN_DATA_PATH = './DATA/'
with open(TRAIN_DATA_PATH + 'mean_and_std_of_training_set.npy', 'rb') as loading_handle:
    x_mean_bar_ch2_red = np.load(loading_handle).item()
    x_mean_bar_ch1_green = np.load(loading_handle).item()
    x_mean_bar_ch0_blue = np.load(loading_handle).item()
    
    x_std_bar_ch2_red = np.load(loading_handle).item()
    x_std_bar_ch1_green = np.load(loading_handle).item()
    x_std_bar_ch0_blue = np.load(loading_handle).item()


# np.dstack stacks 3 h x w arrays -> h x w x 3
RGB_mean_normalize = [x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue]#np.dstack((x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue))
RGB_std_normalize = [x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue]#np.dstack((x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue))

BGR_mean_normalize = [x_mean_bar_ch0_blue,x_mean_bar_ch1_green,x_mean_bar_ch2_red]#np.dstack((x_mean_bar_ch0_blue,x_mean_bar_ch1_green,x_mean_bar_ch2_red))
BGR_std_normalize = [x_std_bar_ch0_blue,x_std_bar_ch1_green,x_std_bar_ch2_red]#np.dstack((x_std_bar_ch0_blue,x_std_bar_ch1_green,x_std_bar_ch2_red))

print(RGB_mean_normalize)
print(RGB_std_normalize)

# BGR Format for cv2
#cv2.imwrite(TRAIN_DATA_PATH + "mean_normalize_train_dataset.png", BGR_mean_normalize) 
#cv2.imwrite(TRAIN_DATA_PATH + "std_normalize_train_dataset.png",  BGR_std_normalize)

############


# Hyper parameters
num_epochs = 20
batchsize = 100
lr = 0.001

EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.003


TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])