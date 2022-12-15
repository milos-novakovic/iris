import torchvision.utils
import cv2
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
#import oyaml as yaml
from collections import OrderedDict
import yaml
#from yaml.loader import SafeLoader
from models import *


from mpl_toolkits.axes_grid1 import ImageGrid


# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x, compose_transforms = None):
    # x dim = (N,C,H,W)
    if compose_transforms == None:
        return x
    
    #np.save('./DATA/RGB_mean.npy', RGB_mean) 
    #RGB_mean = np.load('./DATA/RGB_mean.npy')
    #RGB_std = np.load('./DATA/RGB_std.npy')
    RGB_mean = compose_transforms.transforms[0].mean
    RGB_std = compose_transforms.transforms[0].std
    
    R_mean, G_mean, B_mean = RGB_mean[0], RGB_mean[1], RGB_mean[2]
    R_std, G_std, B_std = RGB_std[0], RGB_std[1], RGB_std[2]
    
    MIN_PIXEL_VALUE, MAX_PIXEL_VALUE = 0,255
    # red chanel of the image
    x[:, 0, :, :] =  R_std * x[:, 0, :, :] + R_mean
    x[:, 0, :, :] = x[:, 0, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    # green chanel of the image
    x[:, 1, :, :] =  G_std * x[:, 1, :, :] + G_mean
    x[:, 1, :, :] = x[:, 1, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    # blue chanel of the image
    x[:, 2, :, :] =  B_std * x[:, 2, :, :] + B_mean
    x[:, 2, :, :] = x[:, 2, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    
    x = np.round(x) #x = np.round(x*255.)
    x = x.int()#astype(int)
    return x

# show/plot top-N worst reconstructed images
def show(original_imgs, reconstructed_imgs, imgs_ids, imgs_losses, savefig_path):
    N,C,H,W = original_imgs.size()
    fig = plt.figure(figsize=(2. * N, 1. * N))#rows,cols
    ncols = 8
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(N//4, ncols),  # creates 2x2 grid of axes
                    axes_pad=0.5,  # pad between axes in inch.
                    )    
    for i,axs in enumerate(grid):  
        im, original_or_reconstructed = None, None
        if i % 2 == 0:
            #axs.set_title(f"({i//2 + 1}/{N}) Org. Test img \n id={imgs_ids[i//2]} loss={imgs_losses[i//2]*1e3:.2f} e-3")
            im = original_imgs[i//2]
            original_or_reconstructed = "Org."
        else:
            #axs.set_title(f"({i//2 + 1}/{N}) Rec. Test img \n id={imgs_ids[i//2]} loss={imgs_losses[i//2]*1e3:.2f} e-3")
            im =  reconstructed_imgs[(i-1)//2]
            original_or_reconstructed = "Rec."
        
        # set title according to the image being original or reconstructed from the Test set (with test image id and its reconstruction loss)
        axs.set_title(f"({i//2 + 1}/{N}) {original_or_reconstructed} Test img \n id={imgs_ids[i//2]} loss={imgs_losses[i//2]*1e3:.2f} e-3")
        axs.imshow(np.transpose(im, (1, 2, 0))) # H,W,C
    
    # save figure and close plotter
    plt.savefig(savefig_path,bbox_inches='tight')
    plt.close()

def visualise_output(images, model, compose_transforms, imgs_ids, imgs_losses, savefig_path, device):

    with torch.no_grad():
        # original images
        # put original mini-batch of images to cpu
        original_images = images.to('cpu') # torch.Size([50, 3, 64, 64])
        
        # reconstructed images
        reconstructed_images = images.to(device)
        model = model.to(device)
        reconstructed_images = model(reconstructed_images)
        if len(reconstructed_images) == 3:
            reconstructed_images = reconstructed_images[1]
        
        # put reconstructed mini-batch of images to cpu
        reconstructed_images = reconstructed_images.to('cpu') # torch.Size([50, 3, 64, 64])        
        
        diff_0_1 = (original_images-reconstructed_images)
        # print statics on difference between original and reconstructed images (0.0-1.0 float range)
        print("The test set difference between original and reconstructed images stats (0.0-1.0 float range):")
        print(f"Size of tensor = {diff_0_1.size()}")
        print(f"Mean of tensor = {diff_0_1.mean()}")
        print(f"Min of tensor = {diff_0_1.min()}")
        print(f"Max of tensor = {diff_0_1.max()}\n")
        
        original_images = to_img(original_images, compose_transforms)
        reconstructed_images = to_img(reconstructed_images, compose_transforms)        
        
        diff_0_255 = (original_images-reconstructed_images)
        # print statics on difference between original and reconstructed images (0-255 int range)
        print("The test set difference between original and reconstructed images stats (0-255 int range):")
        print(f"Size of tensor = {diff_0_255.size()}")
        print(f"Mean of tensor = {diff_0_255.float().mean()}")
        print(f"Min of tensor = {diff_0_255.min()}")
        print(f"Max of tensor = {diff_0_255.max()}\n")
        
        #images = to_img(images, compose_transforms = compose_transforms)
        
        #np_imagegrid_original_images = torchvision.utils.make_grid(tensor = original_images, nrow = 10, padding = 5, pad_value = 255).numpy()
        #np_imagegrid_reconstructed_images = torchvision.utils.make_grid(tensor = reconstructed_images, nrow = 10, padding = 5, pad_value = 255).numpy()
        
        #fig, axs = plt.subplots(1, 10, figsize=(20, 10))
        #plt.imshow(np.transpose(np_imagegrid_original_images, (1, 2, 0))) # H,W,C
        #show(original_images,imgs_ids, imgs_losses)
        
        
        show(original_images,reconstructed_images,imgs_ids, imgs_losses, savefig_path)
        #plt.savefig('./SHOW_IMAGES/org_vs_rec_test_imgs.png',bbox_inches='tight')
        #plt.close()
        
        #fig, axs = plt.subplots(1, 10, figsize=(20, 10))
        #plt.imshow(np.transpose(np_imagegrid_reconstructed_images, (1, 2, 0))) # H,W,C
        #show(reconstructed_images,imgs_ids, imgs_losses)
        
        #show(original_images,reconstructed_images,imgs_ids, imgs_losses)
        #plt.savefig('./SHOW_IMAGES/autoencoder_output_test_50_images.png',bbox_inches='tight')
        #plt.close()

def conv2d_dims(h_in,w_in,k,s,p,d):
    h_out, w_out = None, None
    if len(p) == 2:
        h_out = np.floor( (h_in + 2 * p[0] - d[0] * (k[0] - 1) - 1 ) / s[0] + 1)
        w_out = np.floor( (w_in + 2 * p[1] - d[1] * (k[1] - 1) - 1 ) / s[1] + 1)
    elif len(p) == 4:
        pad_top, pad_bottom, pad_left, pad_right = p[0], p[1], p[2], p[3]
        h_out = np.floor( (h_in + pad_top + pad_bottom - d[0] * (k[0] - 1) - 1 ) / s[0] + 1)
        w_out = np.floor( (w_in + pad_left + pad_right - d[1] * (k[1] - 1) - 1 ) / s[1] + 1)
    return int(h_out), int(w_out)

### Vanilla Autoencoder params begin
def same_padding(h_in, w_in, s, k):
    # SAME padding: This is kind of tricky to understand in the first place because we have to consider two conditions separately as mentioned in the official docs.

    # Let's take input as n_i , output as n_o, padding as p_i, stride as s and kernel size as k (only a single dimension is considered)

    # Case 01: n_i \mod s = 0 :p_i = max(k-s ,0)

    # Case 02: n_i \mod s \neq 0 : p_i = max(k - (n_i\mod s)), 0)

    # p_i is calculated such that the minimum value which can be taken for padding. Since value of p_i is known, value of n_0 can be found using this formula (n_i - k + 2p_i)/2 + 1 = n_0.
    
    #SAME: Apply padding to input (if needed) so that input image gets fully covered by filter and stride you specified. For stride 1, this will ensure that output image size is same as input.
    # p = None
    # if n_i % s == 0:
    #     p = max(k-s ,0)
    # else:
    #     p = max((k - (n_i % s)), 0)
    # return p

    # n_out = np.ceil(float(n_in) / float(s))
    # p = None#int(max((n_out - 1) * s + k - n_in, 0) // 2)
    
    
    # if (n_in % s == 0):
    #     p = max(k - s, 0)
    # else:
    #     p = max(k - (n_in % s), 0)
    # #(2*(output-1) - input - kernel) / stride
    # p = int(p // 2)

    #out_height = np.ceil(float(h_in) / float(s[0]))
    #out_width  = np.ceil(float(h_in) / float(s[1]))

    #The total padding applied along the height and width is computed as:

    if (h_in % s[0] == 0):
        pad_along_height = max(k[0] - s[0], 0)
    else:
        pad_along_height = max(k[0] - (h_in % s[0]), 0)
        
    if (w_in % s[1] == 0):
        pad_along_width = max(k[1] - s[1], 0)
    else:
        pad_along_width = max(k[1] - (w_in % s[1]), 0)
  
    #Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_top, pad_bottom, pad_left, pad_right
# Hyper parameters


# hyperparameters related to images
get_images_hyperparam_value = lambda data_dict, hyperparam_name : \
                            [dict_[hyperparam_name] for dict_ in data_dict['file_info']
                             if hyperparam_name in dict_][0]

images_hyperparam_path = '/home/novakovm/iris/MILOS/milos_config.yaml'
with open(images_hyperparam_path) as f:
    images_hyperparam_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
main_folder_path = get_images_hyperparam_value(images_hyperparam_dict, 'main_folder_path')    

#current_working_absoulte_path = '/home/novakovm/iris/MILOS'
os.chdir(main_folder_path)

# number of images for training and testing datasets
TOTAL_NUMBER_OF_IMAGES      = get_images_hyperparam_value(images_hyperparam_dict, 'TOTAL_NUMBER_OF_IMAGES')
TEST_TOTAL_NUMBER_OF_IMAGES = get_images_hyperparam_value(images_hyperparam_dict, 'TEST_TOTAL_NUMBER_OF_IMAGES')

# load train/val/test dataset percentage take adds up to 100 (percent)
train_dataset_percentage= get_images_hyperparam_value(images_hyperparam_dict, 'train_dataset_percentage')
val_dataset_percentage  = get_images_hyperparam_value(images_hyperparam_dict, 'val_dataset_percentage')
test_dataset_percentage = get_images_hyperparam_value(images_hyperparam_dict, 'test_dataset_percentage')
assert(100 ==train_dataset_percentage+val_dataset_percentage+test_dataset_percentage)

# load train/val/test image ids and check if their number adds up to total number of images
train_shuffled_image_ids= np.load(main_folder_path+'/train_shuffled_image_ids.npy')
val_shuffled_image_ids  = np.load(main_folder_path+'/val_shuffled_image_ids.npy')
test_shuffled_image_ids = np.load(main_folder_path+'/test_shuffled_image_ids.npy')
assert(set(np.concatenate((train_shuffled_image_ids,val_shuffled_image_ids,test_shuffled_image_ids))) == set(np.arange(TOTAL_NUMBER_OF_IMAGES)))

# init config args for train/val/test loaders
args_train, args_val, args_test = {}, {}, {}
args_train['TOTAL_NUMBER_OF_IMAGES'], args_train['image_ids']= TOTAL_NUMBER_OF_IMAGES, train_shuffled_image_ids
args_val['TOTAL_NUMBER_OF_IMAGES'],   args_val['image_ids']  = TOTAL_NUMBER_OF_IMAGES, val_shuffled_image_ids
args_test['TOTAL_NUMBER_OF_IMAGES'], args_test['image_ids']  = TOTAL_NUMBER_OF_IMAGES, test_shuffled_image_ids

# Height, Width, Channel number
H=get_images_hyperparam_value(images_hyperparam_dict, 'H')
W=get_images_hyperparam_value(images_hyperparam_dict, 'W')
C=get_images_hyperparam_value(images_hyperparam_dict, 'C')

# hyperparameters related to training of autoencoder
#models_hyperparam_path = main_folder_path + '/autoencoders_config_64_cont_embed.yaml'
models_hyperparam_path = main_folder_path + '/autoencoders_config_512_cont_embed.yaml'

with open(models_hyperparam_path) as f:
    config_dict = yaml.load(f, Loader=yaml.SafeLoader)

get_config_data = lambda hyper_param_list, hyper_param_name : \
                            [hyper_param for hyper_param in hyper_param_list
                             if hyper_param_name in hyper_param]\
                            [0][hyper_param_name]

NUM_EPOCHS =                            get_config_data(config_dict['training_hyperparams'], 'NUM_EPOCHS')
NUM_WORKERS =                           get_config_data(config_dict['training_hyperparams'], 'NUM_WORKERS') # see what this represents exactly!
LATENT_DIM =                            get_config_data(config_dict['training_hyperparams'], 'LATENT_DIM')# TO DO
USE_PRETRAINED_VANILLA_AUTOENCODER  =   get_config_data(config_dict['training_hyperparams'], 'USE_PRETRAINED_VANILLA_AUTOENCODER')
USE_GPU =                               get_config_data(config_dict['training_hyperparams'], 'USE_GPU')

BATCH_SIZE_TRAIN =                      get_config_data(config_dict['training_hyperparams'], 'BATCH_SIZE_TRAIN')
BATCH_SIZE_VAL =                        get_config_data(config_dict['training_hyperparams'], 'BATCH_SIZE_VAL')
BATCH_SIZE_TEST =                       get_config_data(config_dict['training_hyperparams'], 'BATCH_SIZE_TEST')

LEARNING_RATE =                         get_config_data(config_dict['training_hyperparams'], 'LEARNING_RATE')

TRAIN_DATA_PATH =                       get_config_data(config_dict['training_hyperparams'], 'TRAIN_DATA_PATH')
VAL_DATA_PATH =                         get_config_data(config_dict['training_hyperparams'], 'VAL_DATA_PATH')
TEST_DATA_PATH =                        get_config_data(config_dict['training_hyperparams'], 'TEST_DATA_PATH')

TRAIN_IMAGES_MEAN_FILE_PATH =           get_config_data(config_dict['training_hyperparams'], 'TRAIN_IMAGES_MEAN_FILE_PATH')
TRAIN_IMAGES_STD_FILE_PATH  =           get_config_data(config_dict['training_hyperparams'], 'TRAIN_IMAGES_STD_FILE_PATH')

zero_mean_unit_std_transform = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(256),
    #transforms.ToTensor(),
    transforms.Normalize(mean=np.load(TRAIN_IMAGES_MEAN_FILE_PATH).tolist(),
                         std=np.load(TRAIN_IMAGES_STD_FILE_PATH).tolist() )
    ])

zero_min_one_max_transform = transforms.Compose([
    transforms.Normalize(mean = [0., 0., 0.],
                          std  = [255., 255., 255.])
    ]) # OUTPUT SIGMOID of DNN

minus_one_min_one_max_transform = transforms.Compose([
    transforms.Normalize(mean = [-255./2., -255./2., -255./2.],
                          std  = [255./2., 255./2., 255./2.])
    ]) # OUTPUT TANH of DNN

# Pick one transform that is applied
TRANSFORM_IMG = zero_min_one_max_transform#zero_mean_unit_std_transform # zero_min_one_max_transform

# Train Data & Train data Loader
train_data = CustomImageDataset(args = args_train, root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=NUM_WORKERS)

# Validation Data & Validation data Loader
val_data = CustomImageDataset(args = args_val, root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
val_data_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=BATCH_SIZE_VAL, shuffle=True,  num_workers=NUM_WORKERS)

# Test Data & Test data Loader
test_data = CustomImageDataset(args = args_test, root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = torch.utils.data.DataLoader(dataset = test_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=NUM_WORKERS) 


"""
### ENCODER PARAMS BEGIN
params_encoder= {}

params_encoder['conv1_exists'] = True
params_encoder['in_channels_conv1'] = 3
params_encoder['out_channels_conv1'] = 64
params_encoder['kernel_size_conv1'] = (4,4)
params_encoder['stride_conv1'] = (2,2) # 2,2
params_encoder['padding_conv1'] = (1,1) # 1,1
params_encoder['dilation_conv1']  = (1,1)# 1,1

params_encoder['conv1_H_in'], params_encoder['conv1_W_in'] = H,W # H params_encoder['H_in'] # W params_encoder['W_in']
params_encoder['conv1_H_out'], params_encoder['conv1_W_out'] = conv2d_dims(h_in = params_encoder['conv1_H_in'],w_in = params_encoder['conv1_W_in'],k = params_encoder['kernel_size_conv1'],s = params_encoder['stride_conv1'],p = params_encoder['padding_conv1'],d = params_encoder['dilation_conv1'])
        
# out: c x 14 x 14

# out :
# H_in, W_in even numbers
# H_out = floor( ( H_in + 2 - kernel_size[0]) / 2 + 1)
# W_out = floor( ( W_in + 2 - kernel_size[1]) / 2 + 1)
# default  H_in = W_in = 64
# H_out = floor( ( 64 + 2 - 4) / 2 + 1) = floor(62/2 + 1 ) = 31+1  = 32
# W_out = floor( ( 64 + 2 - 4) / 2 + 1) = floor(62/2 + 1 ) = 31+1  = 32

#self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        
params_encoder['conv2_exists'] = True
params_encoder['out_channels_conv1'] = 64 # 64 = params_encoder['in_channels_conv2']
params_encoder['in_channels_conv2'] = params_encoder['out_channels_conv1']#64
params_encoder['out_channels_conv2'] = 2 * 64 #2 * 64
params_encoder['kernel_size_conv2'] = (4,4) # 4
params_encoder['stride_conv2'] = (2,2) # 2
params_encoder['padding_conv2'] = (1,1) # 1
params_encoder['dilation_conv2']= (1,1)# 1

params_encoder['conv2_H_in'], params_encoder['conv2_W_in'] = params_encoder['conv1_H_out'], params_encoder['conv1_W_out']
params_encoder['conv2_H_out'], params_encoder['conv2_W_out'] = conv2d_dims(h_in = params_encoder['conv2_H_in'],w_in = params_encoder['conv2_W_in'],k = params_encoder['kernel_size_conv2'],s = params_encoder['stride_conv2'],p = params_encoder['padding_conv2'],d = params_encoder['dilation_conv2'])

# out :
# H_in, W_in even numbers
# H_out = floor( ( H_in + 2 - kernel_size[0]) / 2 + 1)
# W_out = floor( ( W_in + 2 - kernel_size[1]) / 2 + 1)
# default  H_in = W_in = 32
# H_out = floor( ( 32 + 2 - 4) / 2 + 1) = floor(30/2 + 1 ) = 15+1  = 16
# W_out = floor( ( 32 + 2 - 4) / 2 + 1) = floor(30/2 + 1 ) = 15+1  = 16  

params_encoder['fc1_exists'] = True
params_encoder['latent_dims'] = LATENT_DIM


### ENCODER PARAMS END
"""
"""
### DECODER PARAMS BEGIN
params_decoder = {}
params_decoder['fc1_exists'] = True
params_decoder['latent_dims'] = LATENT_DIM
params_decoder['in_channels_conv2'] = params_encoder['out_channels_conv2']
params_decoder['conv2_H_in'] = params_encoder['conv2_H_out']
params_decoder['conv2_W_in'] = params_encoder['conv2_W_out']
        
#self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        
params_decoder['conv2_exists'] = params_encoder['conv2_exists']
#params_decoder['in_channels_conv2'],  # 2*64
params_decoder['out_channels_conv2'] = params_encoder['in_channels_conv1']#, # 64
params_decoder['kernel_size_conv2'] = params_encoder['kernel_size_conv2'] # 4
params_decoder['stride_conv2'] = params_encoder['stride_conv2'] # 2
params_decoder['padding_conv2'] = params_encoder['padding_conv2'] # 1
params_decoder['dilation_conv2'] = params_encoder['dilation_conv2']# 1

params_decoder['conv2_H_out'], params_decoder['conv2_W_out'] = conv2d_dims(h_in = params_decoder['conv2_H_in'],w_in = params_decoder['conv2_W_in'],k = params_decoder['kernel_size_conv2'],s = params_decoder['stride_conv2'],p = params_decoder['padding_conv2'],d = params_decoder['dilation_conv2'])
        
#self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
params_decoder['conv1_exists'] = params_encoder['conv1_exists']
#params_decoder['out_channels_conv2'] #,  # 64 = params_decoder['in_channels_conv1']
params_decoder['out_channels_conv1'] = params_encoder['in_channels_conv1']#, # 3
params_decoder['kernel_size_conv1'] = params_encoder['kernel_size_conv1'] # 4
params_decoder['stride_conv1'] = params_encoder['stride_conv1']# 2
params_decoder['padding_conv1'] = params_encoder['padding_conv1']# 1
params_decoder['dilation_conv1'] = params_encoder['dilation_conv1']# 1

params_decoder['conv1_H_in'],params_decoder['conv1_W_in'] = params_encoder['conv1_H_out'],params_encoder['conv1_W_out']

params_decoder['conv1_H_out'],params_decoder['conv1_W_out'] = conv2d_dims(h_in = params_decoder['conv1_H_in'],w_in = params_decoder['conv1_W_in'],k = params_decoder['kernel_size_conv1'],s = params_decoder['stride_conv1'],p = params_decoder['padding_conv1'],d = params_decoder['dilation_conv1'])

### DECODER PARAMS END
"""
"""



params = {}
# conv1 params
params['conv1_H_in'], params['conv1_W_in'], params['stride_conv1'], params['kernel_size_conv1'],params['dilation_conv1'] = \
H, W, (1,1), (3,3), (1,1)
params['in_channels_conv1'], params['out_channels_conv1'] = C,32
#params['padding_conv1'], # calculated
params['padding_conv1_calculated'] = same_padding(  params['conv1_H_in'], 
                                                    params['conv1_W_in'], 
                                                    params['stride_conv1'],
                                                    params['kernel_size_conv1'])
params['padding_conv1'] = 'same'
params['conv1_H_out'],params['conv1_W_out'] = conv2d_dims(h_in = params['conv1_H_in'],
                                                          w_in = params['conv1_W_in'],
                                                          k = params['kernel_size_conv1'],
                                                          s = params['stride_conv1'],
                                                          p = params['padding_conv1_calculated'],
                                                          d = params['dilation_conv1'])
params['maxpool1_H_in'],params['maxpool1_W_in'],params['in_channels_maxpool1'], params['out_channels_maxpool1'] = \
    params['conv1_H_in'], params['conv1_W_in'], params['in_channels_conv1'], 8
params['stride_maxpool1'], params['padding_maxpool1_calculated'], params['kernel_size_maxpool1'],params['dilation_maxpool1'] = \
    (2,2), (0,0,0,0), (2,2), (1,1)



# conv2 params
params['conv2_H_in'], params['conv2_W_in'], params['stride_conv2'], params['kernel_size_conv2'],params['dilation_conv2'] = \
params['conv1_H_out'],params['conv1_W_out'], (2,2), (3,3), (1,1)

params['in_channels_conv2'], params['out_channels_conv2'] = params['out_channels_conv1'],32
#params['padding_conv2'], # calculated
params['padding_conv2_calculated'] = same_padding(  params['conv2_H_in'], 
                                                    params['conv2_W_in'], 
                                                    params['stride_conv2'],
                                                    params['kernel_size_conv2'])
params['padding_conv2'] = 'same'

params['conv2_H_out'],params['conv2_W_out'] = conv2d_dims(h_in = params['conv2_H_in'],
                                                          w_in = params['conv2_W_in'],
                                                          k = params['kernel_size_conv2'],
                                                          s = params['stride_conv2'],
                                                          p = params['padding_conv2_calculated'],
                                                          d = params['dilation_conv2'])
# conv3 params
params['conv3_H_in'], params['conv3_W_in'], params['stride_conv3'], params['kernel_size_conv3'],params['dilation_conv3'] = \
params['conv2_H_out'],params['conv2_W_out'], (1,1), (3,3), (1,1)

params['in_channels_conv3'], params['out_channels_conv3'] = params['out_channels_conv2'],32
#params['padding_conv3'], # calculated
params['padding_conv3_calculated'] = same_padding(  params['conv3_H_in'], 
                                                    params['conv3_W_in'], 
                                                    params['stride_conv3'],
                                                    params['kernel_size_conv3'])
params['padding_conv3'] = 'same'

params['conv3_H_out'],params['conv3_W_out'] = conv2d_dims(h_in = params['conv3_H_in'],
                                                          w_in = params['conv3_W_in'],
                                                          k = params['kernel_size_conv3'],
                                                          s = params['stride_conv3'],
                                                          p = params['padding_conv3_calculated'],
                                                          d = params['dilation_conv3'])


# calculate the dimension of latent space
params['latent_dimension'] = params['out_channels_conv3'] * params['conv3_H_out'] * params['conv3_W_out']

### UpSampling ###
params['upsample1_mode'] = 'nearest'
params['upsample1_scale_factor'] = (2,2)


# conv4 params
params['conv4_H_in'], params['conv4_W_in'], params['stride_conv4'], params['kernel_size_conv4'],params['dilation_conv4'] = \
params['conv3_H_out'] * params['upsample1_scale_factor'][0],params['conv3_W_out'] * params['upsample1_scale_factor'][1], (1,1), (3,3), (1,1)

params['in_channels_conv4'], params['out_channels_conv4'] = params['out_channels_conv3'],32
#params['padding_conv4'], # calculated
params['padding_conv4_calculated'] = same_padding(  params['conv4_H_in'], 
                                                    params['conv4_W_in'], 
                                                    params['stride_conv4'],
                                                    params['kernel_size_conv4'])
params['padding_conv4'] = 'same'

params['conv4_H_out'],params['conv4_W_out'] = conv2d_dims(h_in = params['conv4_H_in'],
                                                          w_in = params['conv4_W_in'],
                                                          k = params['kernel_size_conv4'],
                                                          s = params['stride_conv4'],
                                                          p = params['padding_conv4_calculated'],
                                                          d = params['dilation_conv4'])


# conv5 params
params['conv5_H_in'], params['conv5_W_in'], params['stride_conv5'], params['kernel_size_conv5'],params['dilation_conv5'] = \
params['conv4_H_out'],params['conv4_W_out'], (1,1), (1,1), (1,1)

params['in_channels_conv5'], params['out_channels_conv5'] = params['out_channels_conv4'],params['in_channels_conv1']
#params['padding_conv5'], # calculated
params['padding_conv5_calculated'] = same_padding(  params['conv5_H_in'], 
                                                    params['conv5_W_in'], 
                                                    params['stride_conv5'],
                                                    params['kernel_size_conv5'])
params['padding_conv5'] = 'same'


params['conv5_H_out'],params['conv5_W_out'] = conv2d_dims(h_in = params['conv5_H_in'],
                                                          w_in = params['conv5_W_in'],
                                                          k = params['kernel_size_conv5'],
                                                          s = params['stride_conv5'],
                                                          p = params['padding_conv5_calculated'],
                                                          d = params['dilation_conv5'])

### Vanilla Autoencoder params end
"""

### Vanilla_Autoencoder_v02 params begin

autoencoder_config_params = {}
autoencoder_config_params_file_path = models_hyperparam_path#main_folder_path + '/autoencoders_config.yaml'

with open(autoencoder_config_params_file_path) as f:
    autoencoder_config_params = yaml.load(f, Loader=yaml.SafeLoader)

# all of the params wrapped stored in dict where 
# key is = layer_name
# value is = list of dictionaries that have strucutre like this {feature_value : feature_name}
# e.g. {
#   'conv1':    {'C_in': 3, 'H_in': 64, 'W_in': 64, ...},
#   'maxpool1': {'C_in': 8, 'H_in': 64, 'W_in': 64, ...},
#   ...
#   }

autoencoder_config_params_wrapped_unsorted = {   layer_name:   {
                                                    list(feature_name_feature_value.keys())[0]:
                                                    list(feature_name_feature_value.values())[0]
                                                    for feature_name_feature_value in autoencoder_config_params[layer_name]
                                                    } 
                                                for layer_name in autoencoder_config_params} # if a value is missing default value is None

# get vanilla_autoencoder path direction, name, version and python extension from config file
model_file_path_info={}
model_file_path_info['model_dir_path'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_path']#'/home/novakovm/iris/MILOS/'
model_file_path_info['model_name'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_name']#'vanilla_autoencoder'
model_file_path_info['model_version'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_version']# '_2022_11_20_17_13_14'
model_file_path_info['model_extension'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_extension']#'.py'

# pop the already gotten values from the dict
autoencoder_config_params_wrapped_unsorted.pop('vanilla_autoencoder')
autoencoder_config_params_wrapped_unsorted.pop('training_hyperparams')

# sort layer numbers because the laters have the important ordering defined with Layer_number
sorted_Layer_Numbers = np.sort([autoencoder_config_params_wrapped_unsorted[layer_name]['Layer_Number'] 
                                for layer_name in autoencoder_config_params_wrapped_unsorted]) 

# all of the params wrapped stored in ORDERED dict (OrderedDict) where 
# key of the dict is equal to layer_name
# value of the dict is equal to list of dictionaries that have strucutre like this {feature_value : feature_name}
# e.g. {
#   'conv1':    {'C_in': 3, 'H_in': 64, 'W_in': 64, ...},
#   'maxpool1': {'C_in': 8, 'H_in': 64, 'W_in': 64, ...},
#   ...
#   }
# init empty Ordered Dict (that will have wrapped config params)
autoencoder_config_params_wrapped_sorted = OrderedDict() 

# all of the params unwrapped stored in ORDERED dict (OrderedDict) where 
# key of the dict is equal to the layer_name + "_" + feature_name
# value of the dict is equal to the feature_value
# e.g. {'conv1->C_in': 3, 'conv1->H_in': 64, 'conv1->W_in': 64, 'conv1->C_out': 8, ...}
# init empty Ordered Dict (that will have unwrapped config params)
autoencoder_config_params_unwrapped_sorted = OrderedDict() 

# fill wrapped sorted and unwrapped sorted Ordered Dicts:
for sorted_Layer_Number in sorted_Layer_Numbers:
    
    # find layer name based on the sorted layer number
    layer_name = [layer_name for layer_name in autoencoder_config_params_wrapped_unsorted 
                  if sorted_Layer_Number == autoencoder_config_params_wrapped_unsorted[layer_name]['Layer_Number']][0]
    
    # update AE config params with wrapped structure in sorted manner
    autoencoder_config_params_wrapped_sorted[layer_name] = autoencoder_config_params_wrapped_unsorted[layer_name]
    
    # update AE config params with unwrapped structure in sorted manner
    for feature_name_feature_value in autoencoder_config_params_wrapped_sorted[layer_name]:
        autoencoder_config_params_unwrapped_sorted[layer_name + '->' +feature_name_feature_value] = \
        autoencoder_config_params_wrapped_sorted[layer_name][feature_name_feature_value]

# # VQ VAE params
# vector_quantizer_config_params_wrapped_sorted= {}
# K_bits = 9#14
# #K
# vector_quantizer_config_params_wrapped_sorted['num_embeddings'] = int(2**K_bits)
# #D
# vector_quantizer_config_params_wrapped_sorted['embedding_dim'] = 64#32#64#32
# #commitment loss hyper param
# vector_quantizer_config_params_wrapped_sorted['beta'] = 0.25
# #prior on the embedding space matrix
# vector_quantizer_config_params_wrapped_sorted['E_prior_weight_distribution'] = 'uniform'

# set the training parameters
loss_fn = nn.MSELoss()
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu") 
#model = Vanilla_Autoencoder_v02(autoencoder_config_params_wrapped_sorted)
# model = VQ_VAE(vector_quantizer_config_params_wrapped_sorted = vector_quantizer_config_params_wrapped_sorted,
#                  encoder_config_params_wrapped_sorted = None, 
#                  decoder_config_params_wrapped_sorted = None,
#                  encoder_model = VQ_VAE_Encoder(C_in = 3, C_Conv2d = 64, num_residual_layers = 2),
#                  decoder_model = VQ_VAE_Decoder(C_in = 64, num_residual_layers = 2))

model = vq_vae_implemented_model
#model_name = 'vanilla_autoencoder'
model_name = 'VQ_VAE'
loaders = {'train' : train_data_loader, 'val' : val_data_loader, 'test' : test_data_loader}
optimizer_settings = {'optimization_algorithm':'Adam','lr':LEARNING_RATE}

# create a trainer init arguments
training_args = {}
training_args['NUM_EPOCHS']         = NUM_EPOCHS
training_args['loss_fn']            = loss_fn
training_args['device']             = device
training_args['model']              = model
training_args['model_name']         = model_name
training_args['loaders']            = loaders
training_args['optimizer_settings'] = optimizer_settings #torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-5)
training_args['main_folder_path']   = main_folder_path

# create a trainer object
trainer = Model_Trainer(args=training_args)

if not USE_PRETRAINED_VANILLA_AUTOENCODER:
    # start the training and validation procedure
    trainer.train()
else:
    ##########################    
    # Use a pretrained model #
    ##########################
    current_time_str = "2022_12_15_13_45_59"#"2022_12_15_02_13_36"#"2022_12_03_19_39_08"#'2022_12_02_17_59_16' # 17h 13min 14 sec 20th Nov. 2022
    
    # load model that was trained at newly given current_time_str 
    trainer.load_model(current_time_str = current_time_str, 
                       autoencoder_config_params_wrapped_sorted= autoencoder_config_params_wrapped_sorted)

    # load avg. training loss of the training proceedure for a model that was trained at newly given current_time_str 
    trainer.train_loss_avg = np.load(trainer.main_folder_path
                                     + '/'
                                     + trainer.model_name 
                                     + '_train_loss_avg_'
                                     + trainer.current_time_str + '.npy')
    
    # load avg. validation loss of the validating proceedure for a model that was trained at newly given current_time_str 
    trainer.val_loss_avg = np.load(trainer.main_folder_path
                                    + '/'
                                    + trainer.model_name 
                                    + '_val_loss_avg_'
                                    + trainer.current_time_str + '.npy')
######################    
# Testing the model #
######################
loss_fn = trainer.loss_fn
loss_fn.to(trainer.device)
trainer.test() 

######################    
# Plot train and validation avergae loss across mini-batch across epochs #
# Plot Test Loss for every sample in the Test set #
######################
trainer.plot()

######################    
# Plot test images reconstruction losses
# And for "labels" use different shape features to see  
# which shape features did autoencoder learned the best/worst
# (i.e. what is the easiest/hardest to learn from persepctive of autoencoder)
######################
shape_features_of_interest = [
                            'FILL_NOFILL',
                            'SHAPE_TYPE_SPACE',
                            'X_CENTER_SPACE',
                            'Y_CENTER_SPACE',
                            'COLOR_LIST',
                            'a_CENTER_SPACE',
                            'b_CENTER_SPACE',
                            'alpha_CENTER_SPACE'
                            ]
for shape_feature_of_interest in shape_features_of_interest:
    trainer.scatter_plot_test_images_with_specific_classes(shape_features_of_interest = [shape_feature_of_interest])

######################    
# Plot top-N worst reconstructed test images
# [with their original test images side by side
# and rank them from worst (highest reconstruction loss value)
# to best reconstructed test image]
######################
TOP_WORST_RECONSTRUCTED_TEST_IMAGES = 50
trainer.get_worst_test_samples(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)
trainer.model.eval()
visualise_output(images             = trainer.top_images, 
                 model              = trainer.model,
                 compose_transforms = TRANSFORM_IMG,
                 imgs_ids           = trainer.imgs_ids,
                 imgs_losses        = trainer.imgs_losses,
                 savefig_path       = './SHOW_IMAGES/WORST_RECONSTRUCTED_TEST_IMAGES.png',
                 device = trainer.device)

######################    
# Plot top-N best reconstructed test images
# [with their original test images side by side
# and rank them from best (lowest reconstruction loss value)
# to worst reconstructed test image]
######################
TOP_BEST_RECONSTRUCTED_TEST_IMAGES = 50
trainer.get_best_test_samples(TOP_BEST_RECONSTRUCTED_TEST_IMAGES)
trainer.model.eval()
visualise_output(images             = trainer.top_images, 
                 model              = trainer.model,
                 compose_transforms = TRANSFORM_IMG,
                 imgs_ids           = trainer.imgs_ids,
                 imgs_losses        = trainer.imgs_losses,
                 savefig_path       = './SHOW_IMAGES/BEST_RECONSTRUCTED_TEST_IMAGES.png',
                 device = trainer.device)

debug =0