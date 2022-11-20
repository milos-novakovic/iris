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

def find_mean_std(TOTAL_NUMBER_OF_IMAGES, image_ids_numbers = None, method = 'n'):
    # method = 'n' or method = 'n-1'
    
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

    # only red, green, and blue channel lists for all generated images
    R_imgs,G_imgs,B_imgs = [],[],[]

    for i,image_id in enumerate(image_ids):
        # update empirical estimates for 1st and 2nd centered moments
        image_full_path = './DATA/color_img_' + image_id + '.png'
        
        x = cv2.imread(image_full_path)
        
        R_imgs.append(np.float64(x)[:,:,2])
        G_imgs.append(np.float64(x)[:,:,1])
        B_imgs.append(np.float64(x)[:,:,0])

        # empirically and recursively calculate first moment (E[X] = expectation) per chanel (red, green, blue)
        x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue = \
            x_mean_bar_ch2_red   + (-x_mean_bar_ch2_red   + np.mean(np.float64(x)[:,:,2]))/(i+1), \
            x_mean_bar_ch1_green + (-x_mean_bar_ch1_green + np.mean(np.float64(x)[:,:,1]))/(i+1), \
            x_mean_bar_ch0_blue  + (-x_mean_bar_ch0_blue  + np.mean(np.float64(x)[:,:,0]))/(i+1)
        
        # empirically and recursively calculate second centered moment (E[X^2] = VAR[X] + E[X]^2 = std(X)^2 + E[X]^2) per chanel (red, green, blue)
        x2_mean_bar_ch2_red, x2_mean_bar_ch1_green, x2_mean_bar_ch0_blue = \
            x2_mean_bar_ch2_red     + (-x2_mean_bar_ch2_red   + np.mean(np.float64(x)[:,:,2]**2))/(i+1), \
            x2_mean_bar_ch1_green   + (-x2_mean_bar_ch1_green + np.mean(np.float64(x)[:,:,1]**2))/(i+1), \
            x2_mean_bar_ch0_blue    + (-x2_mean_bar_ch0_blue  + np.mean(np.float64(x)[:,:,0]**2))/(i+1)

    #available
    #x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue
    x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue = None, None, None
    
    if method == 'n-1': # unbiased
        x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue = \
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch2_red   - x_mean_bar_ch2_red**2)),  \
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch1_green - x_mean_bar_ch1_green**2)),\
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch0_blue  - x_mean_bar_ch0_blue**2))  
    
    elif method == 'n': # numpy
        x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue = \
            np.sqrt(x2_mean_bar_ch2_red   - x_mean_bar_ch2_red**2),  \
            np.sqrt(x2_mean_bar_ch1_green - x_mean_bar_ch1_green**2),  \
            np.sqrt(x2_mean_bar_ch0_blue  - x_mean_bar_ch0_blue**2)
    else:
        assert(False)
    

    # two arrays of (3,) size
    RGB_mean = np.array([x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue])#np.dstack((x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue))
    RGB_std = np.array([x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue])#np.dstack((x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue))    
    
    R_imgs,G_imgs,B_imgs = np.array(R_imgs),np.array(G_imgs),np.array(B_imgs)
    RGB_mean_np = np.array([np.mean(R_imgs),np.mean(G_imgs),np.mean(B_imgs)])
    RGB_std_np = np.array([np.std(R_imgs),np.std(G_imgs),np.std(B_imgs)])
    
    # save preprocessing result (i.e. mean and std per changel)
    np.save('./DATA/RGB_mean.npy', RGB_mean) # np.load('./DATA/RGB_mean.npy')
    np.save('./DATA/RGB_std.npy', RGB_std) # np.load('./DATA/RGB_std.npy')
    
    np.save('./DATA/RGB_mean_np.npy', RGB_mean_np) # np.load('./DATA/RGB_mean_np.npy')
    np.save('./DATA/RGB_std_np.npy', RGB_std_np) # np.load('./RGB_std_np/RGB_mean.npy')
    
    #np.sum(np.abs(RGB_mean - RGB_mean_np)) < 1e-14 #  True
    #np.sum(np.abs(RGB_std - RGB_std_np)) < 1e-13 #  True
    return RGB_mean, RGB_std, RGB_mean_np, RGB_std_np



#Step 2 - Take Sample data

#img = Image.open("./DATA/color_img_000.png")
#H,W = img.size

#Step 3 - Convert to tensor

#convert_tensor = transforms.ToTensor()

#tensor_img = convert_tensor(img)

# ch 0 is Red (R)
# ch 1 is Green (G)
# ch 2 is Blue (B)

# BGR Format for cv2
#cv2.imwrite(TRAIN_DATA_PATH + "mean_normalize_train_dataset.png", BGR_mean_normalize) 
#cv2.imwrite(TRAIN_DATA_PATH + "std_normalize_train_dataset.png",  BGR_std_normalize)

############


