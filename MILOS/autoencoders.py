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
# Hyper parameters


# hyperparameters related to images
get_images_hyperparam_value = lambda data_dict, hyperparam_name : [dict_[hyperparam_name] for dict_ in data_dict['file_info'] if hyperparam_name in dict_][0]

images_hyperparam_path = '/home/novakovm/iris/MILOS/milos_config.yaml'
with open(images_hyperparam_path) as f:
    images_hyperparam_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
main_folder_path = get_images_hyperparam_value(images_hyperparam_dict, 'main_folder_path')    
#current_working_absoulte_path = '/home/novakovm/iris/MILOS'
os.chdir(main_folder_path)

# number of images for training and testing datasets
TOTAL_NUMBER_OF_IMAGES = get_images_hyperparam_value(images_hyperparam_dict, 'TOTAL_NUMBER_OF_IMAGES')
TEST_TOTAL_NUMBER_OF_IMAGES = get_images_hyperparam_value(images_hyperparam_dict, 'TEST_TOTAL_NUMBER_OF_IMAGES')

train_dataset_percentage = get_images_hyperparam_value(images_hyperparam_dict, 'train_dataset_percentage')
val_dataset_percentage = get_images_hyperparam_value(images_hyperparam_dict, 'val_dataset_percentage')
test_dataset_percentage = get_images_hyperparam_value(images_hyperparam_dict, 'test_dataset_percentage')
assert(100 ==train_dataset_percentage+val_dataset_percentage+test_dataset_percentage)

train_shuffled_image_ids = np.load(main_folder_path+'/train_shuffled_image_ids.npy')
val_shuffled_image_ids = np.load(main_folder_path+'/val_shuffled_image_ids.npy')
test_shuffled_image_ids = np.load(main_folder_path+'/test_shuffled_image_ids.npy')
assert(set(np.concatenate((train_shuffled_image_ids,val_shuffled_image_ids,test_shuffled_image_ids))) == set(np.arange(TOTAL_NUMBER_OF_IMAGES)))

args_train = {}
args_train['TOTAL_NUMBER_OF_IMAGES'] = TOTAL_NUMBER_OF_IMAGES
args_train['image_ids'] = train_shuffled_image_ids

args_val = {}
args_val['TOTAL_NUMBER_OF_IMAGES'] = TOTAL_NUMBER_OF_IMAGES
args_val['image_ids'] = val_shuffled_image_ids

args_test = {}
args_test['TOTAL_NUMBER_OF_IMAGES'] = TOTAL_NUMBER_OF_IMAGES
args_test['image_ids'] = test_shuffled_image_ids


# Height, Width, Channel number
H=get_images_hyperparam_value(images_hyperparam_dict, 'H')
W=get_images_hyperparam_value(images_hyperparam_dict, 'W')
C=get_images_hyperparam_value(images_hyperparam_dict, 'C')

# hyperparameters related to training of autoencoder
models_hyperparam_path = main_folder_path + '/autoencoders_config.yaml'
with open(models_hyperparam_path) as f:
    models_hyperparam_dict = yaml.load(f, Loader=yaml.SafeLoader)

#models_hyperparam_dict['training_hyperparams']

get_models_hyperparam_value = lambda hyper_param_list, hyper_param_name : [hyper_param for hyper_param in hyper_param_list if hyper_param_name in hyper_param][0][hyper_param_name]
#get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'USE_GPU') #True

NUM_WORKERS = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'NUM_WORKERS') # see what this represents exactly!
LATENT_DIM = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'LATENT_DIM')# TO DO
USE_PRETRAINED_VANILLA_AUTOENCODER  = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'USE_PRETRAINED_VANILLA_AUTOENCODER')
USE_GPU = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'USE_GPU')
#TRAIN_FLAG = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TRAIN_FLAG')

NUM_EPOCHS = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'NUM_EPOCHS')
#BATCH_SIZE = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'BATCH_SIZE')

BATCH_SIZE_TRAIN = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'BATCH_SIZE_TRAIN')
BATCH_SIZE_VAL = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'BATCH_SIZE_VAL')
BATCH_SIZE_TEST = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'BATCH_SIZE_TEST')

LEARNING_RATE = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'LEARNING_RATE')

TRAIN_DATA_PATH = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TRAIN_DATA_PATH')
VAL_DATA_PATH = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'VAL_DATA_PATH')
TEST_DATA_PATH = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TEST_DATA_PATH')

TRAIN_IMAGES_MEAN_FILE_PATH = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TRAIN_IMAGES_MEAN_FILE_PATH')
TRAIN_IMAGES_STD_FILE_PATH  = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TRAIN_IMAGES_STD_FILE_PATH')

zero_mean_unit_std_transform = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(256),
    #transforms.ToTensor(),
    transforms.Normalize(mean=np.load(TRAIN_IMAGES_MEAN_FILE_PATH).tolist(),
                         std=np.load(TRAIN_IMAGES_STD_FILE_PATH).tolist() )
    ])

zero_min_one_max_transform = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(256),
    #transforms.ToTensor(),
    transforms.Normalize(mean = [0., 0., 0.],
                          std  = [255., 255., 255.])
    ])

# Pick one transform that is applied
TRANSFORM_IMG = zero_min_one_max_transform#zero_mean_unit_std_transform # zero_min_one_max_transform

# Train Data & Train data Loader
# Image Folder = A generic data loader where the images are arranged in this way by default

#train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data = CustomImageDataset(args = args_train, root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=NUM_WORKERS)

# Validation Data & Validation data Loader
val_data = CustomImageDataset(args = args_val, root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
val_data_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=BATCH_SIZE_VAL, shuffle=True,  num_workers=NUM_WORKERS)

# Test Data & Test data Loader
#test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data = CustomImageDataset(args = args_test, root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = torch.utils.data.DataLoader(dataset = test_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=NUM_WORKERS) 

# put all loaders into dict
loaders = {
    'train' : train_data_loader,
    'val' : val_data_loader,
    'test' : test_data_loader
}


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
autoencoder_config_params_file_path = main_folder_path + '/autoencoders_config.yaml'

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


model_file_path_info={}
model_file_path_info['model_dir_path'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_path']#'/home/novakovm/iris/MILOS/'
model_file_path_info['model_name'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_name']#'vanilla_autoencoder'
model_file_path_info['model_version'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_version']# '_2022_11_20_17_13_14'
model_file_path_info['model_extension'] = autoencoder_config_params_wrapped_unsorted['vanilla_autoencoder']['vanilla_autoencoder_extension']#'.py'

autoencoder_config_params_wrapped_unsorted.pop('vanilla_autoencoder')
autoencoder_config_params_wrapped_unsorted.pop('training_hyperparams')

sorted_Layer_Numbers = np.sort([autoencoder_config_params_wrapped_unsorted[layer_name]['Layer_Number'] for layer_name in autoencoder_config_params_wrapped_unsorted]) 

# all of the params wrapped stored in ORDERED dict (OrderedDict) where 
# key is = layer_name
# value is = list of dictionaries that have strucutre like this {feature_value : feature_name}
# e.g. {
#   'conv1':    {'C_in': 3, 'H_in': 64, 'W_in': 64, ...},
#   'maxpool1': {'C_in': 8, 'H_in': 64, 'W_in': 64, ...},
#   ...
#   }
autoencoder_config_params_wrapped_sorted = OrderedDict()

# all of the params unwrapped stored in ORDERED dict (OrderedDict) where 
# key is = layer_name + "_" + feature_name
# value is = feature_value
# e.g. {'conv1->C_in': 3, 'conv1->H_in': 64, 'conv1->W_in': 64, 'conv1->C_out': 8, ...}
autoencoder_config_params_unwrapped_sorted = OrderedDict()

# fill wrapped sorted and unwrapped sorted Ordered Dicts:
for sorted_Layer_Number in sorted_Layer_Numbers:
    layer_name = [layer_name for layer_name in autoencoder_config_params_wrapped_unsorted if sorted_Layer_Number == autoencoder_config_params_wrapped_unsorted[layer_name]['Layer_Number']][0]
    # update AE config params with wrapped structure in sorted manner
    autoencoder_config_params_wrapped_sorted[layer_name] = autoencoder_config_params_wrapped_unsorted[layer_name]
    # update AE config params with unwrapped structure in sorted manner
    for feature_name_feature_value in autoencoder_config_params_wrapped_sorted[layer_name]:
        autoencoder_config_params_unwrapped_sorted[layer_name + '->' +feature_name_feature_value] = autoencoder_config_params_wrapped_sorted[layer_name][feature_name_feature_value]

### Vanilla_Autoencoder_v02 params end
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

vanilla_autoencoder_v02 = Vanilla_Autoencoder_v02(autoencoder_config_params_wrapped_sorted=autoencoder_config_params_wrapped_sorted)
vanilla_autoencoder_v02 = vanilla_autoencoder_v02.to(device)
num_params = sum(p.numel() for p in vanilla_autoencoder_v02.parameters() if p.requires_grad)
print('Number of parameters in vanilla AE : %d' % num_params)
optimizer = torch.optim.Adam(params=vanilla_autoencoder_v02.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-5)
#vanilla_autoencoder_v02.train()
print("Vanilla AE v02 model summary is as follows:\n",vanilla_autoencoder_v02)

# device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

# vanilla_autoencoder = Vanilla_Autoencoder(params=params)
# vanilla_autoencoder = vanilla_autoencoder.to(device)
# num_params = sum(p.numel() for p in vanilla_autoencoder.parameters() if p.requires_grad)
# print('Number of parameters in vanilla AE : %d' % num_params)
# optimizer = torch.optim.Adam(params=vanilla_autoencoder.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-5)
# vanilla_autoencoder.train()

#autoencoder = Autoencoder(params_encoder=params_encoder, params_decoder=params_decoder)
#autoencoder = autoencoder.to(device)
#num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
#print('Number of parameters: %d' % num_params)
#optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# set to training mode
#autoencoder.train()


# set the training parameters
training_args = {}
training_args['NUM_EPOCHS'] = NUM_EPOCHS
training_args['loss_fn'] = nn.MSELoss()
training_args['device'] = device #'gpu', i.e.,  device(type='cuda', index=0)
training_args['model'] = vanilla_autoencoder_v02
training_args['model_name'] = 'vanilla_autoencoder'
training_args['loaders'] = loaders# = {'train' : train_data_loader, 'val' : val_data_loader, 'test' : test_data_loader}
training_args['optimizer'] = optimizer #torch.optim.Adam(params=vanilla_autoencoder_v02.parameters(), lr=LEARNING_RATE)
training_args['main_folder_path'] = main_folder_path#'/home/novakovm/iris/MILOS'

# create a trainer object
trainer = Model_Trainer(args=training_args)

if not USE_PRETRAINED_VANILLA_AUTOENCODER:
# start the training and validation procedure
    trainer.train()

"""

MSELoss = nn.MSELoss()
loss_fn = MSELoss #lambda predicted_outputs, outputs: MSELoss(predicted_outputs, outputs)
loss_fn.to(device)

train_loss_avg = []
val_loss_avg = []

training_message_format = lambda current_epoch, \
                                total_nb_epochs, \
                                duration_sec, \
                                train_duration_sec, \
                                val_duration_sec, \
                                batch_size_train,\
                                batch_size_val,\
                                current_avg_train_loss,\
                                current_avg_val_loss, \
                                min_avg_train_loss, \
                                min_avg_val_loss:\
                            f"Epoch {current_epoch+1}/{total_nb_epochs};\n"\
                            +f"Training   Samples Mini-Batch size      = {batch_size_train};\n"\
                            +f"Validation Samples Mini-Batch size      = {batch_size_val};\n"\
                            +f"Total time elapsed in Training Loop     = {train_duration_sec};\n"\
                            +f"Total time elapsed in Validation Loop   = {val_duration_sec};\n"\
                            +f"Total time elapsed                      = {duration_sec};\n"\
                            +f"Curr. Avg. Train Loss across mini-batch = {current_avg_train_loss *1e6 : .1f} e-6;\n"\
                            +f"Curr. Avg. Val   Loss across mini-batch = {current_avg_val_loss *1e6 : .1f} e-6;\n"\
                            +f"Min.  Avg. Train Loss across mini-batch = {min_avg_train_loss *1e6 : .1f} e-6;\n"\
                            +f"Min.  Avg. Val   Loss across mini-batch = {min_avg_val_loss *1e6 : .1f} e-6;\n"



if TRAIN_FLAG and not(USE_PRETRAINED_VANILLA_AUTOENCODER):
    START_TIME_TRAINING = time.time()
    print('Training ...')
    min_train_loss, min_val_loss = np.inf, np.inf
    for epoch in range(NUM_EPOCHS):
        ###################
        # train the model #
        ###################
        start_time_epoch = time.time()
        train_loss_avg.append(0)
        num_batches = 0
        train_loss = 0.0
        vanilla_autoencoder_v02.train()
        
        for image_batch, image_ids_batch  in train_data_loader:
            
            #image_batch.size()
            #torch.Size([128, 3, 64, 64]) = BATCH_SIZE x 3 (RGB) x H x W
            image_batch = image_batch.to(device) # device = device(type='cuda', index=0)
            
            # autoencoder reconstruction # forward pass: compute predicted outputs by passing inputs to the model
            #image_batch_recon = autoencoder(image_batch)
            #image_batch_recon = vanilla_autoencoder(image_batch)
            image_batch_recon = vanilla_autoencoder_v02(image_batch)
            
            # reconstruction error # calculate the batch loss
            #loss = F.mse_loss(image_batch_recon, image_batch)
            loss = loss_fn(image_batch_recon, image_batch)
            
            ## find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            # (backpropagation) backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            
            num_batches += 1
            
        train_loss_avg[-1] /= num_batches
        min_train_loss = np.min([min_train_loss, train_loss_avg[-1]])
        elapsed_time = round(time.time() - start_time_epoch,1)

        
        ######################    
        # validate the model #
        ######################
        start_time_epoch = time.time()
        val_loss_avg.append(0)
        num_batches = 0
        valid_loss = 0.0
        vanilla_autoencoder_v02.eval()
        
        for image_batch, image_ids_batch in val_data_loader:
            
            #image_batch.size()
            #torch.Size([128, 3, 64, 64]) = BATCH_SIZE x 3 (RGB) x H x W
            image_batch = image_batch.to(device) # device = device(type='cuda', index=0)
            
            # autoencoder reconstruction # forward pass: compute predicted outputs by passing inputs to the model
            #image_batch_recon = autoencoder(image_batch)
            #image_batch_recon = vanilla_autoencoder(image_batch)
            image_batch_recon = vanilla_autoencoder_v02(image_batch)
            
            # reconstruction error # calculate the batch loss
            #loss = F.mse_loss(image_batch_recon, image_batch)
            loss = loss_fn(image_batch_recon, image_batch)
            
            # since this is validation of the model there is not (backprogagation and update step size)            
            val_loss_avg[-1] += loss.item()
            num_batches += 1
            
        val_loss_avg[-1] /= num_batches
        min_val_loss = np.min([min_val_loss, val_loss_avg[-1]])
        elapsed_time = round(time.time() - start_time_epoch,1)
        
        if (epoch+1) % 10 == 0:
            total_elapsed_time_seconds = int(time.time() - START_TIME_TRAINING)
            m, s = divmod(total_elapsed_time_seconds, 60)
            h, m = divmod(m, 60)
            print(training_message_format(current_epoch=epoch, 
                                        total_nb_epochs = NUM_EPOCHS,
                                        duration_sec = f"{h}:{m}:{s} hours/mins/secs", 
                                        batch_size_train = BATCH_SIZE_TRAIN,
                                        batch_size_val = BATCH_SIZE_VAL,
                                        current_avg_train_loss = train_loss_avg[-1],
                                        current_avg_val_loss = val_loss_avg[-1],
                                        min_avg_train_loss = min_train_loss,
                                        min_avg_val_loss = min_val_loss),
                    end = "\n-------------------------------------------\n")
        
        #print("\n-------------------------------------------\n")

    current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())) # 2022_11_19_20_11_26

    #train_loss_avg_path = '/home/novakovm/iris/MILOS/autoencoder_train_loss_avg_' + current_time_str + '.npy'
    train_loss_avg_path = main_folder_path + '/vanilla_autoencoder_train_loss_avg_' + current_time_str + '.npy'
    val_loss_avg_path = main_folder_path + '/vanilla_autoencoder_val_loss_avg_' + current_time_str + '.npy'

    train_loss_avg = np.array(train_loss_avg)
    val_loss_avg = np.array(val_loss_avg)
    np.save(train_loss_avg_path,train_loss_avg)
    np.save(val_loss_avg_path,val_loss_avg)
    print(f"Autoencoder Training Loss Average saved here\n{train_loss_avg_path}")
    print(f"Autoencoder Validation Loss Average savedhere\n{val_loss_avg_path}")

    pretrained_autoencoder_path = main_folder_path + '/vanilla_autoencoder_' + current_time_str + '.py'
    #torch.save(autoencoder.state_dict(), pretrained_autoencoder_path)
    #torch.save(vanilla_autoencoder.state_dict(), pretrained_autoencoder_path)


    # SAVING OF vanilla_autoencoder_v02
    #model_file_path_info['model_dir_path'] #'/home/novakovm/iris/MILOS/'
    #model_file_path_info['model_name'] #'vanilla_autoencoder'
    model_file_path_info['model_version']  = current_time_str # '_2022_11_20_17_13_14'
    #model_file_path_info['model_extension'] #'.py'
    model_path =    model_file_path_info['model_dir_path']\
                    + model_file_path_info['model_name'] \
                    + model_file_path_info['model_version'] \
                    + model_file_path_info['model_extension']
    torch.save(vanilla_autoencoder_v02.state_dict(),model_path)
    print(f"Current Trained Model saved at = {model_path}")
"""



"""
vanilla_autoencoder_loaded = None
vanilla_autoencoder_v02_loaded = None

if USE_PRETRAINED_VANILLA_AUTOENCODER:
    current_time_str = "2022_12_03_19_39_08"#'2022_12_02_17_59_16' # 17h 13min 14 sec 20th Nov. 2022
    
    #autoencoder_loaded_path = '/home/novakovm/iris/MILOS/autoencoder_' + current_time_str + '.py'
    #vanilla_autoencoder_loaded_path = '/home/novakovm/iris/MILOS/vanilla_autoencoder_' + current_time_str + '.py'
    vanilla_autoencoder_v02_loaded_path =   model_file_path_info['model_dir_path'] + \
                                            model_file_path_info['model_name'] + \
                                            current_time_str + \
                                            model_file_path_info['model_extension']

    # autoencoder_loaded = Autoencoder(params_encoder=params_encoder, params_decoder=params_decoder)
    # autoencoder_loaded.load_state_dict(torch.load(autoencoder_loaded_path))
    # autoencoder_loaded.eval()

    # vanilla_autoencoder_loaded = Vanilla_Autoencoder(params)
    # vanilla_autoencoder_loaded.load_state_dict(torch.load(vanilla_autoencoder_loaded_path))
    # vanilla_autoencoder_loaded.eval()
    
    vanilla_autoencoder_v02_loaded = Vanilla_Autoencoder_v02(autoencoder_config_params_wrapped_sorted=autoencoder_config_params_wrapped_sorted)
    vanilla_autoencoder_v02_loaded.load_state_dict(torch.load(vanilla_autoencoder_v02_loaded_path))
    vanilla_autoencoder_v02_loaded = vanilla_autoencoder_v02_loaded.to(device=device)
    vanilla_autoencoder_v02_loaded.eval()
"""


if USE_PRETRAINED_VANILLA_AUTOENCODER:
    current_time_str = "2022_12_03_19_39_08"#'2022_12_02_17_59_16' # 17h 13min 14 sec 20th Nov. 2022
    
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

loss_fn = trainer.loss_fn#nn.MSELoss()
loss_fn.to(device)
#train_loss_avg = trainer.train_loss_avg
#val_loss_avg = trainer.val_loss_avg

######################    
# testing the model #
######################
# if USE_PRETRAINED_VANILLA_AUTOENCODER:
#     vanilla_autoencoder_v02_loaded.eval()
# else:
#     vanilla_autoencoder_v02.eval()
   
trainer.test() 
"""
print('Testing ...')
test_loss_avg, num_batches = 0, 0
test_loss = []
test_samples_loss = {} # test_image_tensor: test_loss
test_samples_loss['test_image_id'] = []
test_samples_loss['test_image_rec_loss'] = []

for image_batch, image_id_batch in test_data_loader:
    test_samples_loss['test_image_id'].append(image_id_batch.item())
        
    with torch.no_grad():
        image_batch = image_batch.to(device)
        # autoencoder reconstruction
        image_batch_recon = None
        if USE_PRETRAINED_VANILLA_AUTOENCODER:
            # Pretrained
            #image_batch_recon = vanilla_autoencoder_loaded(image_batch)
            #image_batch_recon = vanilla_autoencoder_loaded(image_batch)
            image_batch_recon = vanilla_autoencoder_v02_loaded(image_batch)
        else:
            # Trained just now
            #image_batch_recon = autoencoder(image_batch)
            #image_batch_recon = vanilla_autoencoder(image_batch)
            image_batch_recon = vanilla_autoencoder_v02(image_batch)

        # reconstruction error
        loss = loss_fn(image_batch_recon, image_batch)

        test_samples_loss['test_image_rec_loss'].append(loss.item())
        test_loss.append(loss.item())
        test_loss_avg += loss.item()
        num_batches += 1
    
test_loss_avg /= num_batches
test_loss = np.array(test_loss)
print('average reconstruction error: %f' % (test_loss_avg))
"""
trainer.plot()


shape_features_of_interest = [  'FILL_NOFILL',
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
"""
# Plot Training and Validation Average Loss per Epoch
if TRAIN_FLAG and not(USE_PRETRAINED_VANILLA_AUTOENCODER):
    plt.figure()
    plt.semilogy(val_loss_avg)
    plt.semilogy(train_loss_avg)
    plt.title(f'Train (Min. = {train_loss_avg.min() *1e3: .2f} e-3) & Validation (Min. = {val_loss_avg.min() *1e3: .2f} e-3) \n Log Loss Avg. per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Avg. Train & Validation Loss')
    plt.legend(['validation','training'])
    plt.grid()
    plt.savefig(main_folder_path + '/semilog_train_val_loss_per_epoch.png')
    plt.close()

# Plot Test Loss for every sample in the Test set
plt.figure()
plt.stem(test_loss)
plt.title(f'Test Loss per minibatch (Avg. = {test_loss.mean()*1e3 : .2f} e-3)')
plt.xlabel('Mini-batch')
plt.ylabel('Testing Loss')
plt.savefig(main_folder_path + '/testing_loss_per_image_in_minibatch.png')
plt.close()

"""
TOP_WORST_RECONSTRUCTED_TEST_IMAGES = 50
trainer.get_worst_test_samples(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)

"""
# Visualization of top worst reconstructed test images (i.e. where autoencoder fails) 
df_test_samples_loss = pd.DataFrame(trainer.test_samples_loss)
df_test_samples_loss = df_test_samples_loss.sort_values('test_image_rec_loss',ascending=False)\
                                           .reset_index(drop=True)
#pick top 50 worst reconstructed images
TOP_WORST_RECONSTRUCTED_TEST_IMAGES = 100
df_worst_reconstructed_test_images = df_test_samples_loss.head(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)
print(f"pick top {TOP_WORST_RECONSTRUCTED_TEST_IMAGES} worst reconstructed images\n", df_worst_reconstructed_test_images.to_string())

top_images = []
imgs_ids , imgs_losses = [], []
for worst_reconstructed_test_image_id, worst_reconstructed_test_image_loss in zip(df_worst_reconstructed_test_images['test_image_id'], df_worst_reconstructed_test_images['test_image_rec_loss']):
    # find the test image index when you have test image id in the test_data.image_ids tha
    worst_reconstructed_test_image_id_index = np.where(test_data.image_ids == worst_reconstructed_test_image_id)[0][0]
    
    # get the actual image as well as the image_id
    image, image_id = test_data[worst_reconstructed_test_image_id_index]
    
    # save the test image (tensor)
    top_images.append(image)
    
    # save the test image id
    imgs_ids.append(image_id)
    
    # save the test image reconstruction error (i.e. loss value)
    imgs_losses.append(worst_reconstructed_test_image_loss)

# saved top_images are list of tensor, so cast to a tensor with torch.stack() function
#torch.Size(TOP_WORST_RECONSTRUCTED_TEST_IMAGES, C, H, W)
top_images = torch.stack(top_images) 
"""


trainer.model.eval()
"""
# Put model into evaluation mode
if USE_PRETRAINED_VANILLA_AUTOENCODER:
    # Pretrained
    #vanilla_autoencoder_loaded.eval()
    vanilla_autoencoder_v02_loaded.eval()
else:
    # Trained just now
    #vanilla_autoencoder.eval()
    vanilla_autoencoder_v02.eval()
"""
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

def visualise_output(images, model, compose_transforms, imgs_ids, imgs_losses, savefig_path):

    with torch.no_grad():
        # original images
        # put original mini-batch of images to cpu
        original_images = images.to('cpu') # torch.Size([50, 3, 64, 64])
        
        # reconstructed images
        reconstructed_images = images.to(device)
        model = model.to(device)
        reconstructed_images = model(reconstructed_images)
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

# Reconstruct and visualise the images using the autoencoder



visualise_output(images             = trainer.top_images, 
                 model              = trainer.model,
                 compose_transforms = TRANSFORM_IMG,
                 imgs_ids           = trainer.imgs_ids,
                 imgs_losses        = trainer.imgs_losses,
                 savefig_path       = './SHOW_IMAGES/org_vs_rec_test_imgs.png')


"""
if USE_PRETRAINED_VANILLA_AUTOENCODER:
    # Pretrained
    #visualise_output(top_images, vanilla_autoencoder_loaded, compose_transforms = TRANSFORM_IMG)
    visualise_output(top_images, vanilla_autoencoder_v02_loaded, compose_transforms = TRANSFORM_IMG, imgs_ids = imgs_ids, imgs_losses=imgs_losses, savefig_path = './SHOW_IMAGES/org_vs_rec_test_imgs.png')
else:
    # Trained just now
    #visualise_output(top_images, vanilla_autoencoder, compose_transforms = TRANSFORM_IMG)
    visualise_output(top_images, vanilla_autoencoder_v02, compose_transforms = TRANSFORM_IMG, imgs_ids = imgs_ids, imgs_losses=imgs_losses, savefig_path = './SHOW_IMAGES/org_vs_rec_test_imgs.png')
"""
debug =0