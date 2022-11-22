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
# Hyper parameters
current_working_absoulte_path = '/home/novakovm/iris/MILOS'
os.chdir(current_working_absoulte_path)

# hyperparameters related to images
get_images_hyperparam_value = lambda data_dict, hyperparam_name : [dict_[hyperparam_name] for dict_ in data_dict['file_info'] if hyperparam_name in dict_][0]

images_hyperparam_path = '/home/novakovm/iris/MILOS/milos_config.yaml'
with open(images_hyperparam_path) as f:
    images_hyperparam_dict = yaml.load(f, Loader=yaml.SafeLoader)

# number of images for training and testing datasets
TOTAL_NUMBER_OF_IMAGES = get_images_hyperparam_value(images_hyperparam_dict, 'TOTAL_NUMBER_OF_IMAGES')
TEST_TOTAL_NUMBER_OF_IMAGES = get_images_hyperparam_value(images_hyperparam_dict, 'TEST_TOTAL_NUMBER_OF_IMAGES')
args_train = {}
args_train['TOTAL_NUMBER_OF_IMAGES'] = TOTAL_NUMBER_OF_IMAGES
args_test = {}
args_test['TOTAL_NUMBER_OF_IMAGES'] = TEST_TOTAL_NUMBER_OF_IMAGES

# Height, Width, Channel number
H=get_images_hyperparam_value(images_hyperparam_dict, 'H')
W=get_images_hyperparam_value(images_hyperparam_dict, 'W')
C=get_images_hyperparam_value(images_hyperparam_dict, 'C')

# hyperparameters related to training of autoencoder
models_hyperparam_path = '/home/novakovm/iris/MILOS/autoencoders_config.yaml'
with open(models_hyperparam_path) as f:
    models_hyperparam_dict = yaml.load(f, Loader=yaml.SafeLoader)

#models_hyperparam_dict['training_hyperparams']

get_models_hyperparam_value = lambda hyper_param_list, hyper_param_name : [hyper_param for hyper_param in hyper_param_list if hyper_param_name in hyper_param][0][hyper_param_name]
#get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'USE_GPU') #True

NUM_WORKERS = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'NUM_WORKERS') # see what this represents exactly!
LATENT_DIM = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'LATENT_DIM')# TO DO
USE_PRETRAINED_VANILLA_AUTOENCODER  = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'USE_PRETRAINED_VANILLA_AUTOENCODER')
USE_GPU = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'USE_GPU')
TRAIN_FLAG = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TRAIN_FLAG')

NUM_EPOCHS = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'NUM_EPOCHS')
BATCH_SIZE = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'BATCH_SIZE')
LEARNING_RATE = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'LEARNING_RATE')

TRAIN_DATA_PATH = get_models_hyperparam_value(models_hyperparam_dict['training_hyperparams'], 'TRAIN_DATA_PATH')
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
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)

# Test Data & Test data Loader
#test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data = CustomImageDataset(args = args_test, root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = torch.utils.data.DataLoader(dataset = test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) 

# Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=TRAIN_DATA_PATH,
#                                            batch_size=batchsize,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=TEST_DATA_PATH,
#                                           batch_size=batchsize,
#                                           shuffle=False)


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
autoencoder_config_params_file_path = '/home/novakovm/iris/MILOS/autoencoders_config.yaml'

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
vanilla_autoencoder_v02.train()
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

loss_fn = nn.MSELoss()
loss_fn.to(device)

train_loss_avg = []
if TRAIN_FLAG:
    print('Training ...')
    for epoch in range(NUM_EPOCHS):
        start_time_epoch = time.time()
        train_loss_avg.append(0)
        num_batches = 0
        
        for image_batch in train_data_loader:
            
            #image_batch.size()
            #torch.Size([128, 3, 64, 64]) = BATCH_SIZE x 3 (RGB) x H x W
            image_batch = image_batch.to(device) # device = device(type='cuda', index=0)
            
            # autoencoder reconstruction
            #image_batch_recon = autoencoder(image_batch)
            #image_batch_recon = vanilla_autoencoder(image_batch)
            image_batch_recon = vanilla_autoencoder_v02(image_batch)
            
            
            # reconstruction error
            #loss = F.mse_loss(image_batch_recon, image_batch)
            loss = loss_fn(image_batch_recon, image_batch)
            
            # init grad params
            optimizer.zero_grad()
            
            # backpropagation
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, NUM_EPOCHS, train_loss_avg[-1]))#2h
        print(f'{epoch}th epoch took {int(time.time() - start_time_epoch)} seconds. ')
        # TO DO:
        ### NICE TO SEE IN THE TRAINING LOOP like in tensor flow
        # Epoch 48/50
        # 196/196 [==============================] - 32s 164ms/step - loss: 2.3840e-04 - accuracy: 0.9044 - val_loss: 2.6590e-04 - val_accuracy: 0.8891
        # Epoch 49/50
        # 196/196 [==============================] - 32s 164ms/step - loss: 2.3283e-04 - accuracy: 0.9032 - val_loss: 3.2471e-04 - val_accuracy: 0.9093
        # Epoch 50/50
        # 196/196 [==============================] - 32s 166ms/step - loss: 2.4793e-04 - accuracy: 0.9034 - val_loss: 4.8787e-04 - val_accuracy: 0.8481
        # 313/313 [==============================] - 1s 3ms/step
        ###
        
        
        
    
    current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())) # 2022_11_19_20_11_26
    
    #train_loss_avg_path = '/home/novakovm/iris/MILOS/autoencoder_train_loss_avg_' + current_time_str + '.npy'
    train_loss_avg_path = '/home/novakovm/iris/MILOS/vanilla_autoencoder_train_loss_avg_' + current_time_str + '.npy'
    print(f"Autoencoder Training Loss Average = {train_loss_avg}")
    
    train_loss_avg = np.array(train_loss_avg)
    np.save(train_loss_avg_path,train_loss_avg)
 
    
    pretrained_autoencoder_path = '/home/novakovm/iris/MILOS/vanilla_autoencoder_' + current_time_str + '.py'
    #torch.save(autoencoder.state_dict(), pretrained_autoencoder_path)
    #torch.save(vanilla_autoencoder.state_dict(), pretrained_autoencoder_path)
    
    
    # SAVING OF vanilla_autoencoder_v02
    #model_file_path_info['model_dir_path'] #'/home/novakovm/iris/MILOS/'
    #model_file_path_info['model_name'] #'vanilla_autoencoder'
    model_file_path_info['model_version']  = current_time_str # '_2022_11_20_17_13_14'
    #model_file_path_info['model_extension'] #'.py'
    torch.save(vanilla_autoencoder_v02.state_dict(), model_file_path_info['model_dir_path'] + model_file_path_info['model_name'] + model_file_path_info['model_version'] + model_file_path_info['model_extension'])

# plt.ion()

# fig = plt.figure()
# plt.plot(train_loss_avg)
# plt.xlabel('Epochs')
# plt.ylabel('Reconstruction error')
# plt.show()


vanilla_autoencoder_loaded = None
vanilla_autoencoder_v02_loaded = None

if USE_PRETRAINED_VANILLA_AUTOENCODER:
    current_time_str = '2022_11_20_17_13_14' # 17h 13min 14 sec 20th Nov. 2022
    
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
    vanilla_autoencoder_v02_loaded.eval()


print('Testing ...')
test_loss_avg, num_batches = 0, 0
test_loss = []
for image_batch in test_data_loader:
    
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
        loss = F.mse_loss(image_batch_recon, image_batch)

        
        test_loss.append(loss.item())
        test_loss_avg += loss.item()
        num_batches += 1
    
test_loss_avg /= num_batches
test_loss = np.array(test_loss)
print('average reconstruction error: %f' % (test_loss_avg))



plt.figure()
plt.plot(train_loss_avg)
#plt.savefig('/home/novakovm/iris/MILOS/training_loss_per_epoch_'+current_time_str+'.png')

plt.title(f'Training Loss Avg. per epoch (Min. = {train_loss_avg.min()*1e3 : .2f}e-3)')
plt.xlabel('Epochs')
plt.ylabel('Avg. Training Loss')
plt.savefig('/home/novakovm/iris/MILOS/training_loss_per_epoch.png')
plt.close()

plt.figure()
plt.plot(test_loss)
#plt.savefig('/home/novakovm/iris/MILOS/training_loss_per_epoch_'+current_time_str+'.png')
plt.title(f'Test Loss per minibatch (Avg. = {test_loss.mean()*1e3 : .2f}e-3)')
plt.xlabel('Mini-batch')
plt.ylabel('Testing Loss')
plt.savefig('/home/novakovm/iris/MILOS/testing_loss_per_image_in_minibatch.png')
plt.close()

#plt.ion()


# Put model into evaluation mode
#autoencoder.eval()

if USE_PRETRAINED_VANILLA_AUTOENCODER:
    # Pretrained
    #vanilla_autoencoder_loaded.eval()
    vanilla_autoencoder_v02_loaded.eval()
else:
    # Trained just now
    #vanilla_autoencoder.eval()
    vanilla_autoencoder_v02.eval()
    


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

# def show_image(img, compose_transforms, path = './SHOW_IMAGES/show_image.png'):
    
#     images = to_img(img, compose_transforms = compose_transforms) # C = size(0), H = size(1), W = size(2)
#     np_imagegrid = torchvision.utils.make_grid(images, nrow = 10, padding = 5, pad_value = 255).numpy()
#     plt.imshow(np.transpose(np_imagegrid, (1, 2, 0))) # H,W,C
#     plt.savefig(path)
#     plt.close()

def visualise_output(images, model, compose_transforms):

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
        
        # print statics on test set original (real) images (0.0-1.0 float range)
        print("The test set original (real) images stats (0.0-1.0 float range):")
        print(f"Size of tensor = {original_images.size()}")
        print(f"Mean of tensor = {original_images.mean()}")
        print(f"Min of tensor = {original_images.min()}")
        print(f"Max of tensor = {original_images.max()}\n")
        
        
        # print statics on reconstructed images (0.0-1.0 float range)
        print("The test set reconstructed images stats (0.0-1.0 float range):")
        print(f"Size of tensor = {reconstructed_images.size()}")
        print(f"Mean of tensor = {reconstructed_images.mean()}")
        print(f"Min of tensor = {reconstructed_images.min()}")
        print(f"Max of tensor = {reconstructed_images.max()}\n")
        
        
        diff_0_1 = (original_images-reconstructed_images)
        # print statics on difference between original and reconstructed images (0.0-1.0 float range)
        print("The test set difference between original and reconstructed images stats (0.0-1.0 float range):")
        print(f"Size of tensor = {diff_0_1.size()}")
        print(f"Mean of tensor = {diff_0_1.mean()}")
        print(f"Min of tensor = {diff_0_1.min()}")
        print(f"Max of tensor = {diff_0_1.max()}\n")
        
        
        original_images = to_img(original_images, compose_transforms)
        reconstructed_images = to_img(reconstructed_images, compose_transforms)
        
        
        # print statics on test set original (real) images (0-255 int range)
        print("The test set original (real) images stats (0-255 int range):")
        print(f"Size of tensor = {original_images.size()}")
        print(f"Mean of tensor = {original_images.float().mean()}")
        print(f"Min of tensor = {original_images.min()}")
        print(f"Max of tensor = {original_images.max()}\n")
        
        
        # print statics on reconstructed images (0-255 int range)
        print("The test set reconstructed images stats (0-255 int range):")
        print(f"Size of tensor = {reconstructed_images.size()}")
        print(f"Mean of tensor = {reconstructed_images.float().mean()}")
        print(f"Min of tensor = {reconstructed_images.min()}")
        print(f"Max of tensor = {reconstructed_images.max()}\n")
        
        
        diff_0_255 = (original_images-reconstructed_images)
        # print statics on difference between original and reconstructed images (0-255 int range)
        print("The test set difference between original and reconstructed images stats (0-255 int range):")
        print(f"Size of tensor = {diff_0_255.size()}")
        print(f"Mean of tensor = {diff_0_255.float().mean()}")
        print(f"Min of tensor = {diff_0_255.min()}")
        print(f"Max of tensor = {diff_0_255.max()}\n")
        
        #images = to_img(images, compose_transforms = compose_transforms)
        
        np_imagegrid_original_images = torchvision.utils.make_grid(tensor = original_images, nrow = 10, padding = 5, pad_value = 255).numpy()
        np_imagegrid_reconstructed_images = torchvision.utils.make_grid(tensor = reconstructed_images, nrow = 10, padding = 5, pad_value = 255).numpy()
        
        plt.imshow(np.transpose(np_imagegrid_original_images, (1, 2, 0))) # H,W,C
        plt.savefig('./SHOW_IMAGES/original_test_50_images.png')
        plt.close()
        
        plt.imshow(np.transpose(np_imagegrid_reconstructed_images, (1, 2, 0))) # H,W,C
        plt.savefig('./SHOW_IMAGES/autoencoder_output_test_50_images.png')
        plt.close()

some_test_images = iter(test_data_loader).next() # torch.Size([128, 3, 64, 64])

PICK_TOP_N_IMAGES = 50
top_images = some_test_images[:PICK_TOP_N_IMAGES, :, :, :]
# First visualise the original images
# print('Original images')
# show_image(top_images,compose_transforms = TRANSFORM_IMG, path = './SHOW_IMAGES/show_image.png')
#plt.show()

# Reconstruct and visualise the images using the autoencoder
# print('Autoencoder reconstruction:')
#visualise_output(top_images, autoencoder, compose_transforms= )

if USE_PRETRAINED_VANILLA_AUTOENCODER:
    # Pretrained
    #visualise_output(top_images, vanilla_autoencoder_loaded, compose_transforms = TRANSFORM_IMG)
    visualise_output(top_images, vanilla_autoencoder_v02_loaded, compose_transforms = TRANSFORM_IMG)
else:
    # Trained just now
    #visualise_output(top_images, vanilla_autoencoder, compose_transforms = TRANSFORM_IMG)
    visualise_output(top_images, vanilla_autoencoder_v02, compose_transforms = TRANSFORM_IMG)
debug =0

#'./data/MNIST_AE_pretrained/my_autoencoder.pth'

# this is how the autoencoder parameters can be saved:
#torch.save(autoencoder.state_dict(), pretrained_autoencoder_path) # OBAVEZNO!