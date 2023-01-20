import yaml
from yaml.loader import SafeLoader
import itertools
import glob
from collections import OrderedDict
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
# for functionvisualize_model_as_graph_image
# taken from https://github.com/mert-kurttutan/torchview
from torchview import draw_graph
import graphviz # conda install graphviz

from helper_functions import to_img

# projections from high dim. space to a lower one
import umap
from sklearn.decomposition import PCA
#import sklearn

import os
import pandas as pd
#from torchvision.io import read_image

#from vq_vae_implementation import *
from mpl_toolkits import mplot3d
import imageio
from matplotlib.ticker import FormatStrFormatter

from helper_functions import get_hyperparam_from_config_file

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, args, root = False, transform=None):#, target_transform=False):
        #self.img_labels = pd.read_csv(annotations_file)
        self.root = root # './DATA/' = '/home/novakovm/iris/MILOS/DATA/'
        self.transform = transform
        self.TOTAL_NUMBER_OF_IMAGES = args['TOTAL_NUMBER_OF_IMAGES'] # 2048 for test and val; 12'288 for train
        self.image_ids = args['image_ids']# 3314, 2151, 12030, 32, ...
        
    def __len__(self):
        return len(self.image_ids)#self.TOTAL_NUMBER_OF_IMAGES

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.root + 'color_img_' + str(image_id).zfill(len(str(self.TOTAL_NUMBER_OF_IMAGES))) + '.png'#os.path.join(self.root, self.img_labels.iloc[idx, 0])
        #image = torchvision.io.read_image(img_path).float() # .double() = torch.float64 and  .float() = torch.float32
        image = torchvision.io.read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB).float() # .double() = torch.float64 and  .float() = torch.float32
        
        if self.transform != None:
            image = self.transform(image)
        return image, image_id#, label


class Encoder(nn.Module):
    def __init__(self, params_encoder):
        super(Encoder, self).__init__()
        
        #params_encoder = {}
        
        #C,H,W = params_encoder['C'], , params_encoder['W']
        #capacity
        
        # H_out = floor( ( H_in + 2 * padding[0] - dilatation[0] * (kernel_size[0] - 1) - 1 ) / stride[0] + 1)
        # W_out = floor( ( W_in + 2 * padding[1] - dilatation[1] * (kernel_size[1] - 1) - 1 ) / stride[1] + 1)
        
        if params_encoder['conv1_exists']:
            self.conv1 = nn.Conv2d( in_channels=params_encoder['in_channels_conv1'], #3
                                    out_channels=params_encoder['out_channels_conv1'], #64
                                    kernel_size=params_encoder['kernel_size_conv1'], # 4,4
                                    stride=params_encoder['stride_conv1'], # 2,2
                                    padding=params_encoder['padding_conv1'], # 1,1
                                    dilation= params_encoder['dilation_conv1']) # 1,1
            self.conv1_H_out = params_encoder['conv1_H_out']
            self.conv1_W_out = params_encoder['conv1_W_out']
        
        # out: c x 14 x 14
        
        # out :
        # H_in, W_in even numbers
        # H_out = floor( ( H_in + 2 - kernel_size[0]) / 2 + 1)
        # W_out = floor( ( W_in + 2 - kernel_size[1]) / 2 + 1)
        # default  H_in = W_in = 64
        # H_out = floor( ( 64 + 2 - 4) / 2 + 1) = floor(62/2 + 1 ) = 31+1  = 32
        # W_out = floor( ( 64 + 2 - 4) / 2 + 1) = floor(62/2 + 1 ) = 31+1  = 32
        
        #self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        
        if params_encoder['conv2_exists']:
            self.conv2 = nn.Conv2d( in_channels=params_encoder['out_channels_conv1'], # 64 = params_encoder['in_channels_conv2']
                                    out_channels=params_encoder['out_channels_conv2'], #2 * 64
                                    kernel_size=params_encoder['kernel_size_conv2'], # 4
                                    stride=params_encoder['stride_conv2'], # 2
                                    padding=params_encoder['padding_conv2'], # 1
                                    dilation= params_encoder['dilation_conv2']) # 1
            self.conv2_H_out = params_encoder['conv2_H_out']
            self.conv2_W_out = params_encoder['conv2_W_out']
            #params_encoder['conv2_H_out'] = self.conv2_H_out
            #params_encoder['conv2_W_out'] = self.conv2_W_out
        # out :
        # H_in, W_in even numbers
        # H_out = floor( ( H_in + 2 - kernel_size[0]) / 2 + 1)
        # W_out = floor( ( W_in + 2 - kernel_size[1]) / 2 + 1)
        # default  H_in = W_in = 32
        # H_out = floor( ( 32 + 2 - 4) / 2 + 1) = floor(30/2 + 1 ) = 15+1  = 16
        # W_out = floor( ( 32 + 2 - 4) / 2 + 1) = floor(30/2 + 1 ) = 15+1  = 16
        
        
        if params_encoder['fc1_exists']:
            self.fc1 = nn.Linear(   in_features =   params_encoder['out_channels_conv2'] \
                                                    * self.conv2_H_out \
                                                    * self.conv2_W_out,
                                    out_features= params_encoder['latent_dims'])
            
    def forward(self, x):
        # x.size() = torch.Size([128, 3, 64, 64])
        x = F.relu(self.conv1(x))
        # x.size() = torch.Size([128, 64, 32, 32])
        x = F.relu(self.conv2(x))
        # x.size() = torch.Size([128, 128, 16, 16])
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        # x.size() = torch.Size([128, 32768])
        x = self.fc1(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, params_decoder):
        super(Decoder, self).__init__()
        #c = capacity
        
        if params_decoder['fc1_exists']:
            self.fc1 = nn.Linear(   in_features =  params_decoder['latent_dims'], 
                                    out_features=  params_decoder['in_channels_conv2'] \
                                                    * params_decoder['conv2_H_in'] \
                                                    * params_decoder['conv2_W_in'])
        
        self.C_in_conv2 = params_decoder['in_channels_conv2']
        self.H_in_conv2 = params_decoder['conv2_H_in']
        self.W_in_conv2 = params_decoder['conv2_W_in']
        
        #self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        
        if params_decoder['conv2_exists']:
            self.conv2 = nn.ConvTranspose2d( in_channels=params_decoder['in_channels_conv2'],  # 2*64
                                    out_channels=params_decoder['out_channels_conv2'], # 64
                                    kernel_size=params_decoder['kernel_size_conv2'], # 4
                                    stride=params_decoder['stride_conv2'], # 2
                                    padding=params_decoder['padding_conv2'], # 1
                                    dilation= params_decoder['dilation_conv2']) # 1
            
            self.conv2_H_out = params_decoder['conv2_H_out']
            self.conv2_W_out = params_decoder['conv2_W_out']
        
        #self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        if params_decoder['conv1_exists']:
            self.conv1 = nn.ConvTranspose2d( in_channels=params_decoder['out_channels_conv2'],  # 64 = params_decoder['in_channels_conv1']
                                    out_channels=params_decoder['out_channels_conv1'], # 3
                                    kernel_size=params_decoder['kernel_size_conv1'], # 4
                                    stride=params_decoder['stride_conv1'], # 2
                                    padding=params_decoder['padding_conv1'], # 1
                                    dilation= params_decoder['dilation_conv1']) # 1
            
            self.conv1_H_out = params_decoder['conv1_H_out']
            self.conv1_W_out = params_decoder['conv1_W_out']        
            
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), self.C_in_conv2, self.H_in_conv2, self.W_in_conv2) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x)) # last layer before output is tanh, since the images are normalized and 0-centered ???
        return x
    
class Autoencoder(nn.Module):
    def __init__(self,params_encoder, params_decoder):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(params_encoder)
        self.decoder = Decoder(params_decoder)
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon
    
    

class Vanilla_Autoencoder(nn.Module):
    def __init__(self,params):
        super(Vanilla_Autoencoder, self).__init__()
        #self.encoder = Encoder(params_encoder)
        #self.decoder = Decoder(params_decoder)
        
        ### Torch ###

        ### CONV 1 ###
        #model.add(Conv2D(filters = 32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
        # model.add(Conv2D(filters = 32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
        self.conv1 = nn.Conv2d( in_channels=params['in_channels_conv1'], #3
                                out_channels=params['out_channels_conv1'], #32
                                 kernel_size=params['kernel_size_conv1'], # 3,3
                                stride=params['stride_conv1'], # 1,1
                                #padding=params['padding_conv1'], # calculated
                                dilation= params['dilation_conv1']) # 1,1 - default

        # calculated= padding order =  pad_top, pad_bottom, pad_left, pad_right
        self.conv1_padding = params['padding_conv1_calculated']
        
        self.conv1_H_out = params['conv1_H_out']	#32
        self.conv1_W_out = params['conv1_W_out']	#32
         #activation ReLU
         #x = F.relu(self.conv1(x))

        #model.add(BatchNormalization())     # 32x32x32
        self.bn1 = nn.BatchNorm2d(num_features = params['out_channels_conv1'])#nn.BatchNorm2d(num_features = C_out_conv1)

        ### CONV 2 ###
        #model.add(Conv2D(filters = 32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
        self.conv2 = nn.Conv2d( in_channels=params['in_channels_conv2'], #32
                                out_channels=params['out_channels_conv2'], #32
                                kernel_size=params['kernel_size_conv2'], # 3,3
                                stride=params['stride_conv2'], # 2,2
                                #padding=params['padding_conv2'], # calculated
                                dilation= params['dilation_conv2']) # 1,1 - default
        
        # calculated= padding order =  pad_top, pad_bottom, pad_left, pad_right
        self.conv2_padding = params['padding_conv2_calculated']
        
        self.conv2_H_out = params['conv2_H_out']            
        self.conv2_W_out = params['conv2_W_out']

        #activation ReLU
        #x = F.relu(self.conv2(x))

        ### CONV 3 ###
        #model.add(Conv2D(filters = 32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
        self.conv3 = nn.Conv2d( in_channels=params['in_channels_conv3'], #32
                                out_channels=params['out_channels_conv3'], #32
                                kernel_size=params['kernel_size_conv3'], # 3,3
                                stride=params['stride_conv3'], # 1,1
                                #padding=params['padding_conv3'], # calculated
                                dilation= params['dilation_conv3']) # 1,1 - default

        # calculated= padding order =  pad_top, pad_bottom, pad_left, pad_right
        self.conv3_padding = params['padding_conv3_calculated']

        self.conv3_H_out = params['conv3_H_out']            
        self.conv3_W_out = params['conv3_W_out']

        #activation ReLU
        #x = F.relu(self.conv3(x))

        #model.add(BatchNormalization())     # 16x16x32
        self.bn2 = nn.BatchNorm2d(num_features = params['out_channels_conv3'])#nn.BatchNorm2d(num_features = C_out_conv3)
        
        # calculate the dimension of latent space
        self.latent_dimension = params['latent_dimension']
        ### UpSampling ###
        #tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest') # 16x16x32
        
        
        #params['upsample1_mode'] = 'nearest'
        #params['upsample1_scale_factor'] = (2,2)
        self.upsample1 = torch.nn.Upsample(size=None, scale_factor=params['upsample1_scale_factor'], mode=params['upsample1_mode'], align_corners=None, recompute_scale_factor=None)

        ### CONV 4 ###
        #model.add(Conv2D(filters = 32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
        self.conv4 = nn.Conv2d( in_channels=params['in_channels_conv4'], #32
                                out_channels=params['out_channels_conv4'], #32
                                kernel_size=params['kernel_size_conv4'], # 3,3
                                stride=params['stride_conv4'], # 1,1
                                #padding=params['padding_conv4'], # calculated
                                dilation= params['dilation_conv4']) # 1,1 - default

        # calculated= padding order =  pad_top, pad_bottom, pad_left, pad_right
        self.conv4_padding = params['padding_conv4_calculated']

        self.conv4_H_out = params['conv4_H_out']            
        self.conv4_W_out = params['conv4_W_out']

        #activation ReLU
        #x = F.relu(self.conv4(x))


        #model.add(BatchNormalization())     # 32x32x32
        self.bn3 = nn.BatchNorm2d(num_features = params['out_channels_conv4'])#nn.BatchNorm2d(num_features = C_out_conv4)


        ### CONV 5 ###
        #model.add(Conv2D(filters = 3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3
        self.conv5 = nn.Conv2d( in_channels=params['in_channels_conv5'], #32
                                out_channels=params['out_channels_conv5'], #3
                                kernel_size=params['kernel_size_conv5'], # 1,1
                                stride=params['stride_conv5'], # 1,1
                                #padding=params['padding_conv5'], # calculated
                                dilation= params['dilation_conv5']) # 1,1 - default

        # calculated= padding order =  pad_top, pad_bottom, pad_left, pad_right
        self.conv5_padding = params['padding_conv5_calculated']

        self.conv5_H_out = params['conv5_H_out']            
        self.conv5_W_out = params['conv5_W_out']

        #activation SIGMOID
        #x = F.sigmoid(self.conv5(x))

        # Training
        # model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
        
    def forward(self, x):
        # encoder part
        
        # H : pad_top, pad_bottom,
        # W : pad_left, pad_right
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv1_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        
        x = F.pad(x, padding, "constant", 0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv2_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        
        x = F.pad(x, padding, "constant", 0)
        x = self.conv2(x)
        x = F.relu(x)
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv3_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, padding, "constant", 0)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)
        
        #latent_tensor = x
        
        # decoder part
        x = self.upsample1(x)
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv4_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, padding, "constant", 0)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)
        

        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv5_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, padding, "constant", 0)
        
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x
        
        #latent = self.encoder(x)
        #x_recon = self.decoder(latent)
        #return x_recon
        

class Vanilla_Autoencoder_v02(nn.Module):
    def __init__(self,autoencoder_config_params_wrapped_sorted):
        super(Vanilla_Autoencoder_v02, self).__init__()
        self.autoencoder_config_params_wrapped_sorted = autoencoder_config_params_wrapped_sorted
        
        params = autoencoder_config_params_wrapped_sorted
        self.layers = {}
        # 'vanilla_autoencoder_vanilla_autoencoder_path': '/home/novakovm/iris/MILOS/',
        # 'vanilla_autoencoder_vanilla_autoencoder_name': 'vanilla_autoencoder',
        # 'vanilla_autoencoder_vanilla_autoencoder_version': '_2022_11_20_17_13_14',
        # 'vanilla_autoencoder_vanilla_autoencoder_extension': '.py'
        
        self.sequential_model = torch.nn.Sequential()
        
        self.latent_space_dim = 0
        #self.conv_paddings = {}
        
        for param_name in params:
            param_value_dict = params[param_name]
            if param_name == 'vanilla_autoencoder':
                continue            
            
            if param_name[:len('conv')] == 'conv':
                # keys in the param_value_dict =
                # 'C_in'
                # 'H_in'
                # 'W_in'
                # 'C_out'
                # 'H_out'
                # 'W_out'
                # 'Embedding_Dim'
                # 'Layer_Name'
                # 'conv1'
                # 'Stride_H'
                # 'Stride_W'
                # 'Padding_H_top'
                # 'Padding_H_bottom'
                # 'Padding_W_left'
                # 'Padding_W_right'
                # 'kernel_num'
                # 'Kernel_H'
                # 'Kernel_W'
                # 'Dilation_H'
                # 'Dilation_W'
                
                # First we calculate the padding
                padding = ( param_value_dict['Padding_W_left'],
                            param_value_dict['Padding_W_right'],
                            param_value_dict['Padding_H_top'],
                            param_value_dict['Padding_H_bottom'])
                self.sequential_model.add_module(param_name + '_padding', nn.ZeroPad2d(padding))
                
                # Second we do 2D convolution
                self.sequential_model.add_module(param_name, torch.nn.Conv2d(
                                                                in_channels = param_value_dict['C_in'],
                                                                out_channels= param_value_dict['C_out'],
                                                                kernel_size = (param_value_dict['Kernel_H'], param_value_dict['Kernel_W']),
                                                                stride      = (param_value_dict['Stride_H'], param_value_dict['Stride_W']),
                                                                #padding = calculated already!
                                                                dilation    = (param_value_dict['Dilation_H'], param_value_dict['Dilation_W'])
                                                                ))                
            elif param_name[:len('maxpool')] == 'maxpool':
                # First we calculate the padding
                padding = ( param_value_dict['Padding_W_left'],
                            param_value_dict['Padding_W_right'],
                            param_value_dict['Padding_H_top'],
                            param_value_dict['Padding_H_bottom'])
                self.sequential_model.add_module(param_name + '_padding', nn.ZeroPad2d(padding))
                
                # Second we do 2D maxpooling
                self.sequential_model.add_module(param_name, torch.nn.MaxPool2d(
                                                                kernel_size = (param_value_dict['Kernel_H'], param_value_dict['Kernel_W']),
                                                                stride      = (param_value_dict['Stride_H'], param_value_dict['Stride_W']),
                                                                #padding = calculated already!
                                                                dilation    = (param_value_dict['Dilation_H'], param_value_dict['Dilation_W']),
                                                                return_indices=False,
                                                                ceil_mode=False
                                                                ))
                
            elif param_name[:len('ReLU')] == 'ReLU':
                # Apply ReLU as an activation function
                self.sequential_model.add_module(param_name, torch.nn.ReLU(inplace=False))
                
            elif param_name[:len('bn')] == 'bn':
                # Apply Batch Normalization
                self.sequential_model.add_module(param_name, torch.nn.BatchNorm2d(num_features = param_value_dict['C_in']))
                
            elif param_name[:len('UpSample')] == 'UpSample':
                # Apply UpSampling to restore the original size of an image
                self.sequential_model.add_module(param_name, torch.nn.Upsample(
                                                                            size=None, 
                                                                            scale_factor=(param_value_dict['Stride_H'], param_value_dict['Stride_W']), 
                                                                            mode='nearest', # NEAREST MODE IS HARD-CODED # TO DO (MAKE IT SUPPORT OTHER MODES OF UPSAMPLING)
                                                                            align_corners=None,
                                                                            recompute_scale_factor=None
                                                                        ))
                
            elif param_name[:len('sigmoid')] == 'sigmoid':
                # Apply Sigmoid as an activation function
                self.sequential_model.add_module(param_name, torch.nn.Sigmoid())
            
            else:
                assert(False, f'Unknown config parameter name = {param_name}.')
            
            # update latent space dimension
            self.latent_space_dim = min(self.latent_space_dim , param_value_dict['Embedding_Dim'])
            
        
    def forward(self, x):
        #latent = self.encoder(x)
        #x_recon = self.decoder(latent)
        x_recon = self.sequential_model(x)
        return x_recon


def get_sequential_modules(autoencoder_config_params_wrapped_sorted) -> torch.nn.Sequential:
    params = autoencoder_config_params_wrapped_sorted
    sequential = torch.nn.Sequential()
    
    for param_name in params:
        param_value_dict = params[param_name]
        if param_name == 'vanilla_autoencoder':
            continue            

        if param_name[:len('conv')] == 'conv':
            # keys in the param_value_dict =
            # 'C_in'
            # 'H_in'
            # 'W_in'
            # 'C_out'
            # 'H_out'
            # 'W_out'
            # 'Embedding_Dim'
            # 'Layer_Name'
            # 'conv1'
            # 'Stride_H'
            # 'Stride_W'
            # 'Padding_H_top'
            # 'Padding_H_bottom'
            # 'Padding_W_left'
            # 'Padding_W_right'
            # 'kernel_num'
            # 'Kernel_H'
            # 'Kernel_W'
            # 'Dilation_H'
            # 'Dilation_W'
            
            # First we calculate the padding
            padding = ( param_value_dict['Padding_W_left'],
                        param_value_dict['Padding_W_right'],
                        param_value_dict['Padding_H_top'],
                        param_value_dict['Padding_H_bottom'])
            sequential.add_module(param_name + '_padding', nn.ZeroPad2d(padding))
            
            # Second we do 2D convolution
            sequential.add_module(param_name, torch.nn.Conv2d(
                                                    in_channels = param_value_dict['C_in'],
                                                    out_channels= param_value_dict['C_out'],
                                                    kernel_size = (param_value_dict['Kernel_H'], param_value_dict['Kernel_W']),
                                                    stride      = (param_value_dict['Stride_H'], param_value_dict['Stride_W']),
                                                    #padding = calculated already!
                                                    dilation    = (param_value_dict['Dilation_H'], param_value_dict['Dilation_W'])
                                                    ))
        elif param_name[:len('trans_conv')] == 'trans_conv':            
            # First we calculate the padding
            padding = ( param_value_dict['Padding_W_left'],
                        param_value_dict['Padding_W_right'],
                        param_value_dict['Padding_H_top'],
                        param_value_dict['Padding_H_bottom'])
            sequential.add_module(param_name + '_padding', nn.ZeroPad2d(padding))
            
            # Second we do 2D convolution
            sequential.add_module(param_name, torch.nn.Conv2d(
                                                    in_channels = param_value_dict['C_in'],
                                                    out_channels= param_value_dict['C_out'],
                                                    kernel_size = (param_value_dict['Kernel_H'], param_value_dict['Kernel_W']),
                                                    stride      = (param_value_dict['Stride_H'], param_value_dict['Stride_W']),
                                                    #padding = calculated already!
                                                    dilation    = (param_value_dict['Dilation_H'], param_value_dict['Dilation_W'])
                                                    ))                
        elif param_name[:len('maxpool')] == 'maxpool':
            # First we calculate the padding
            padding = ( param_value_dict['Padding_W_left'],
                        param_value_dict['Padding_W_right'],
                        param_value_dict['Padding_H_top'],
                        param_value_dict['Padding_H_bottom'])
            sequential.add_module(param_name + '_padding', nn.ZeroPad2d(padding))
            
            # Second we do 2D Transpose Convolution
            sequential.add_module(param_name, torch.nn.ConvTranspose2d(
                                                    in_channels = param_value_dict['C_in'],
                                                    out_channels= param_value_dict['C_out'],
                                                    kernel_size = (param_value_dict['Kernel_H'], param_value_dict['Kernel_W']),
                                                    stride      = (param_value_dict['Stride_H'], param_value_dict['Stride_W']),
                                                    #padding = calculated already!
                                                    dilation    = (param_value_dict['Dilation_H'], param_value_dict['Dilation_W'])
                                                    ))
            
        elif param_name[:len('ReLU')] == 'ReLU':
            # Apply ReLU as an activation function
            sequential.add_module(param_name, torch.nn.ReLU(inplace=False))
            
        elif param_name[:len('bn')] == 'bn':
            # Apply Batch Normalization
            sequential.add_module(param_name, torch.nn.BatchNorm2d(num_features = param_value_dict['C_in']))
            
        elif param_name[:len('UpSample')] == 'UpSample':
            # Apply UpSampling to restore the original size of an image
            sequential.add_module(param_name, torch.nn.Upsample(
                                                                size=None, 
                                                                scale_factor=(param_value_dict['Stride_H'], param_value_dict['Stride_W']), 
                                                                mode='nearest', # NEAREST MODE IS HARD-CODED # TO DO (MAKE IT SUPPORT OTHER MODES OF UPSAMPLING)
                                                                align_corners=None,
                                                                recompute_scale_factor=None
                                                                ))
            
        elif param_name[:len('sigmoid')] == 'sigmoid':
            # Apply Sigmoid as an activation function
            sequential.add_module(param_name, torch.nn.Sigmoid())

        else:
            assert(False, f'Unknown config parameter name = {param_name}.')

    return sequential

class VQ_VAE(nn.Module):
    def __init__(self,
                 vector_quantizer_config_params_wrapped_sorted = None,
                 encoder_config_params_wrapped_sorted = None, 
                 decoder_config_params_wrapped_sorted = None,
                 encoder_model = None,
                 decoder_model = None):#autoencoder_config_params_wrapped_sorted):
        super(VQ_VAE, self).__init__()
        #self.autoencoder_config_params_wrapped_sorted = autoencoder_config_params_wrapped_sorted
        self.encoder_config_params_wrapped_sorted           = encoder_config_params_wrapped_sorted
        self.decoder_config_params_wrapped_sorted           = decoder_config_params_wrapped_sorted
        self.vector_quantizer_config_params_wrapped_sorted  = vector_quantizer_config_params_wrapped_sorted
        
        if encoder_config_params_wrapped_sorted != None and decoder_config_params_wrapped_sorted != None and encoder_model == None and decoder_model == None:
            self.encoder = get_sequential_modules(self.encoder_config_params_wrapped_sorted)
            self.decoder = get_sequential_modules(self.decoder_config_params_wrapped_sorted)
        
        if encoder_config_params_wrapped_sorted == None and decoder_config_params_wrapped_sorted == None and encoder_model != None and decoder_model != None:
            self.encoder = encoder_model
            self.decoder = decoder_model
        
        
        self.K    = self.vector_quantizer_config_params_wrapped_sorted['num_embeddings'] # 512
        self.D    = self.vector_quantizer_config_params_wrapped_sorted['embedding_dim'] # 64
        self.beta = self.vector_quantizer_config_params_wrapped_sorted['beta'] # 0.25
        self.E_prior_weight_distribution = vector_quantizer_config_params_wrapped_sorted['E_prior_weight_distribution']#'uniform'
        
        self.E = nn.Embedding(self.K, self.D)
        if self.E_prior_weight_distribution == 'uniform':
            self.E.weight.data.uniform_(-1/self.K, 1/self.K)
        #else:
            #assert(False, f"{self.E_prior_weight_distribution} is not implemented as the prior on the weights of embedded matrix space E.")
        
    def forward(self, x):
        # encoder pass
        encoder_output = self.encoder(x) # [B, C_e, H_e, W_e]
        B, C_e, H_e, W_e = encoder_output.shape
        N_e = B * H_e * W_e
        
        # encoder output reorganized into the matrix 
        # [B, C_e, H_e, W_e] -> [B, H_e, W_e, C_e] -> [B * H_e * W_e, C_e]
        # flatten encoder output into an array of C_e - sized real vectors
        Ze_mat = encoder_output.permute(0, 2, 3, 1).contiguous().view(N_e, C_e)
        
        # calculate distance matrix D
        # D_(i,j) is defined as l2-distance =
        # || Ze_mat_i - E_j ||^2
        D = (
            torch.sum(Ze_mat ** 2, dim = 1, keepdim=True)
            + torch.sum(self.E.weight ** 2, dim = 1, keepdim=False) # or + torch.sum(self.E.weight ** 2, dim = 1, keepdim=True).t()
            - 2 * torch.matmul(Ze_mat, self.E.weight.t())
            )
        
        # try also this one
        # D = (Ze_mat ** 2) @ torch.ones(C_e, self.K) + torch.ones(N_e, C_e) @ (self.E.weight ** 2).t() - 2 * Ze_mat @ self.E.weight.t()
        
        # calculate the encoding indices that are closses in the l2-sense (accroding to the distance matrix D)
        EI = torch.argmin(D, dim=1).unsqueeze(1)
        
        # calculate the one hot encoding indices that are closses in the l2-sense (accroding to the distance matrix D)
        # init with zero matrix of dimensions N_e and K (device same as the output of an encoder)
        OEI = torch.zeros(N_e, self.K, device=encoder_output.device)
        # fill in the one hot encoding based on the encoding indicies EI
        OEI.scatter_(1,#go over each row (axis =1) and put on the column index by EI the fill-in value
                     EI,#the index vector that specifices in which column to put the fill-in value for every row of the index vector
                     1) #the fill-in value
        
        # get the quantized vectors from the embedding space (i.e. code book) E,
        # which index is defined as a one-hot encoding in the OEI matrix
        Zq_mat = torch.matmul(OEI, self.E.weight) # (N_e x D) <- (N_e x K) @ (K x D)
        
        # unflatten Zq_mat matrix to the right decoder input tensor size
        decoder_input = Zq_mat.view(encoder_output.shape)
        
        # Loss calculation
        vq_loss         = F.mse_loss(encoder_output.detach(), decoder_input)
        commitment_loss = F.mse_loss(encoder_output,          decoder_input.detach())
        self.commitment_and_vq_loss = vq_loss + self.beta * commitment_loss
        
        # implementation of the straight-through gradient estimation of mapping from Ze_mat to Zq_mat
        decoder_input = encoder_output + (decoder_input - encoder_output).detach()
        
        # decoder pass
        x_recon = self.decoder(decoder_input)
        
        return x_recon
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 C_in,
                 C_out,
                 C_mid):
        '''
        DocString comment here
        '''
        super(ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
                                    nn.ReLU(True),
                                    nn.Conv2d(  in_channels=C_in,
                                                out_channels=C_mid,
                                                kernel_size=3, 
                                                stride=1,
                                                padding=1,
                                                dilation = 1,
                                                bias=False),
                                    # with k=3, s=1, p=1, d = 1
                                    # H_out = floor( (H_in + p_top + p_bottom - d * (k-1) -1) / s + 1)
                                    # H_out = floor( (H_in + 1 + 1 - 1 * (3-1) -1) / 1 + 1)
                                    # H_out = H_in
                                    # i.e., H_out = H_in and W_out = W_in
                                    nn.ReLU(True),
                                    nn.Conv2d(  in_channels=C_mid,
                                                out_channels=C_out,
                                                kernel_size=1, 
                                                stride=1,
                                                padding=0,
                                                dilation = 1,
                                                bias=False)
                                    )
        # i.e., H_out = H_in and W_out = W_in
    
    def forward(self, x):
        return x + self.residual_block(x)


class ResidualStack(nn.Module):
    def __init__(self, 
                 C_in,
                 C_out,
                 C_mid,
                 num_residual_layers):
        '''
        DocString comment here
        '''
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([ResidualBlock(C_in,C_out,C_mid) for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        x = F.relu(x)
        return x

class VQ_VAE_Encoder(nn.Module):
    def __init__(self,
                 C_in=3,
                 C_Conv2d=32,
                 num_residual_layers=2):
        super(VQ_VAE_Encoder, self).__init__()
        '''
        DocString comment here
        '''
        self.conv1 = nn.Conv2d(in_channels=C_in,
                                 out_channels=C_Conv2d,
                                 kernel_size=4,
                                 stride=2, 
                                 padding=1,
                                 dilation=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features = C_Conv2d)
        self.conv2 = nn.Conv2d(in_channels=C_Conv2d,
                                 out_channels=C_Conv2d,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1,
                                 dilation=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features = C_Conv2d)
        self.conv3 = nn.Conv2d(in_channels=C_Conv2d,
                                 out_channels=C_Conv2d,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dilation=1)
        self.residual_stack = ResidualStack(C_in = C_Conv2d,
                                             C_out = C_Conv2d,
                                             C_mid = C_Conv2d,
                                             num_residual_layers = num_residual_layers)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.residual_stack(x)
        return x
    
class VQ_VAE_Decoder(nn.Module):
    def __init__(self, 
                 C_in = 32,
                 num_residual_layers = 2):
        super(VQ_VAE_Decoder, self).__init__()
        '''
        DocString comment here
        '''
        self.conv1 = nn.Conv2d(in_channels=C_in,
                                 out_channels=C_in,
                                 kernel_size=3, 
                                 stride=1,
                                 padding=1,
                                 dilation=1)
        
        self.residual_stack = ResidualStack(C_in = C_in, C_out = C_in, C_mid = C_in, num_residual_layers = num_residual_layers)
        
        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=C_in, 
                                                out_channels=C_in,
                                                kernel_size=4, 
                                                stride=2,
                                                padding=1)
        
        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=C_in, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, 
                                                padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans_1(x))    
        x = self.conv_trans_2(x)
        x = torch.sigmoid(x)    
        return x
    
    
class My_VQ_VAE_Encoder_Decoder(nn.Module):
    def __init__(self,
                 C_in=3,
                 C_Conv2d=32,
                 num_residual_layers=2):
        super(My_VQ_VAE_Encoder_Decoder, self).__init__()
        '''
        DocString comment here
        '''
        self.conv1 = nn.Conv2d(in_channels=C_in,
                                 out_channels=C_Conv2d,
                                 kernel_size=4,
                                 stride=2, 
                                 padding=1,
                                 dilation=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features = C_Conv2d)
        self.conv2 = nn.Conv2d(in_channels=C_Conv2d,
                                 out_channels=C_Conv2d,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1,
                                 dilation=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features = C_Conv2d)
        self.conv3 = nn.Conv2d(in_channels=C_Conv2d,
                                 out_channels=C_Conv2d,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dilation=1)
        self.residual_stack = ResidualStack(C_in = C_Conv2d,
                                             C_out = C_Conv2d,
                                             C_mid = C_Conv2d,
                                             num_residual_layers = num_residual_layers)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.residual_stack(x)
        return x
    
class My_VQ_VAE_Decoder(nn.Module):
    def __init__(self, 
                 C_in = 32,
                 num_residual_layers = 2):
        super(My_VQ_VAE_Decoder, self).__init__()
        '''
        DocString comment here
        '''
        self.conv1 = nn.Conv2d(in_channels=C_in,
                                 out_channels=C_in,
                                 kernel_size=3, 
                                 stride=1,
                                 padding=1,
                                 dilation=1)
        
        self.residual_stack = ResidualStack(C_in = C_in, C_out = C_in, C_mid = C_in, num_residual_layers = num_residual_layers)
        
        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=C_in, 
                                                out_channels=C_in,
                                                kernel_size=4, 
                                                stride=2,
                                                padding=1)
        
        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=C_in, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, 
                                                padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans_1(x))    
        x = self.conv_trans_2(x)
        x = torch.sigmoid(x)    
        return x
# batch_size = 256
# num_training_updates = 15000

# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# embedding_dim = 64
# num_embeddings = 512

# commitment_cost = 0.25

# decay = 0.99

# learning_rate = 1e-3

class Model_Trainer:
    def __init__(self, args) -> None:
        self.NUM_EPOCHS =       args['NUM_EPOCHS']#1000
        self.loss_fn =          args['loss_fn']#nn.MSELoss()
        self.device =           args['device']#'gpu', i.e.,  device(type='cuda', index=0)
        self.model =            args['model']#vanilla_autoencoder_v02
        self.model_name =       args['model_name']#'vanilla_autoencoder'
        self.loaders =          args['loaders'] # loaders = {'train' : train_data_loader, 'val' : val_data_loader, 'test' : test_data_loader}
        self.optimizer_settings=args['optimizer_settings'] #torch.optim.Adam(params=vanilla_autoencoder_v02.parameters(), lr=LEARNING_RATE)
        self.main_folder_path = args['main_folder_path']#'/home/novakovm/iris/MILOS'
        self.train_data_variance=args['train_data_variance']#float32 number represents the variance of all the training images across all chanels and pixels
        self.PCA_decomp_in_every_epochs = args['PCA_decomp_in_every_epochs']
        self.logger_path = args['logger_path']
        self.config_path = args['config_path']
        
        self.epoch_ids_PCA = []
        # create and init self.optimizer
        if self.optimizer_settings['optimization_algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.optimizer_settings['lr'])
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.NUM_EPOCHS//3 , 2*(self.NUM_EPOCHS//3)], gamma=self.optimizer_settings['lr'] * 0.1)
            #self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.optimizer_settings['lr'])
            
        elif self.optimizer_settings['optimization_algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(params = self.model.parameters(), lr = self.optimizer_settings['lr'])
            
        else:
            assert(False, f"There are no optimizers called {self.optimizer_settings['optimization_algorithm']}.")

    def get_intermediate_training_stats_str(  self,\
                                            current_epoch, \
                                            start_time_training,\
                                            ) -> str:
        train_duration_sec      = self.train_duration_per_epoch_seconds[current_epoch]
        val_duration_sec        = self.val_duration_per_epoch_seconds[current_epoch]
        
        current_avg_train_loss  = self.train_loss_avg[-1]
        current_avg_val_loss    = self.val_loss_avg[-1]
        
        # training duration to str
        m, s = divmod(train_duration_sec, 60)
        h, m = divmod(m, 60)
        train_duration_sec_str = f"{h}:{m}:{s} h/m/s"
        
        # validation duration to str
        m, s = divmod(val_duration_sec, 60)
        h, m = divmod(m, 60)
        val_duration_sec_str = f"{h}:{m}:{s} h/m/s"
        
        # total time elapsed from the beginning of training
        m, s = divmod(int(time.time() - start_time_training), 60)
        h, m = divmod(m, 60)
        duration_sec_str = f"{h}:{m}:{s} h/m/s"
        
        # if we use EMA update of the codebook then we do not have || Z_e.detach() - Z_q ||^2 loss term (i.e. self.train_multiple_losses_avg['VQ_codebook_loss'] = self.val_multiple_losses_avg['VQ_codebook_loss'] = 0)
        non_reconstruction_loss_str = 'beta' if self.model.args_VQ['use_EMA'] else '(1+beta)'
        #+f"Total elapsed time in Training Epoch    = {train_duration_sec_str};\n"\
        #+f"Total elapsed time in Validation Epoch  = {val_duration_sec_str};\n"\
        current_local_time = time.strftime("%H:%M:%S %d.%m.%Y", time.localtime(time.time()))
        intermediate_training_stats : str = \
                f"[{current_local_time}]\n"\
                +f"Epoch {current_epoch+1}/{self.NUM_EPOCHS};\n"\
                +f"Training/Validation   Samples Mini-Batch size      = {self.loaders['train'].batch_size}/{self.loaders['val'].batch_size};\n"\
                +f"Total elapsed time from begining        = {duration_sec_str};\n"\
                +f"Curr. Avg. Train Loss across Mini-Batch = {current_avg_train_loss *1e6 : .1f} e-6; = "\
                                    +f"(1/var)*||X-X_r||^2 = {self.train_multiple_losses_avg['reconstruction_loss'][-1] *1e6 : .1f} e-6 = {self.train_multiple_losses_avg['reconstruction_loss'][-1]/current_avg_train_loss*100:.1f} %; "\
                                    +f"{non_reconstruction_loss_str}*||Z_e-Z_q||^2 = {(self.train_multiple_losses_avg['VQ_codebook_loss'][-1] +  self.model.args_VQ['beta'] * self.train_multiple_losses_avg['commitment_loss'][-1]) *1e6 : .1f} e-6 = {(self.train_multiple_losses_avg['VQ_codebook_loss'][-1] +  self.model.args_VQ['beta'] * self.train_multiple_losses_avg['commitment_loss'][-1])/current_avg_train_loss*100:.1f} %)\n"\
                +f"Curr. Avg. Val   Loss across Mini-Batch = {current_avg_val_loss *1e6 : .1f} e-6; = "\
                                    +f"(1/var)*||X-X_r||^2 = {self.val_multiple_losses_avg['reconstruction_loss'][-1] *1e6 : .1f} e-6 = {self.val_multiple_losses_avg['reconstruction_loss'][-1]/current_avg_val_loss*100:.1f} %; "\
                                    +f"{non_reconstruction_loss_str}*||Z_e-Z_q||^2 = {(self.val_multiple_losses_avg['VQ_codebook_loss'][-1] +  self.model.args_VQ['beta'] * self.val_multiple_losses_avg['commitment_loss'][-1]) *1e6 : .1f} e-6 = {(self.val_multiple_losses_avg['VQ_codebook_loss'][-1] +  self.model.args_VQ['beta'] * self.val_multiple_losses_avg['commitment_loss'][-1])/current_avg_val_loss*100:.1f} %)\n" \
                +f"Min.  Avg. Train Loss across Mini-Batch = {self.min_train_loss *1e6 : .1f} e-6; \n"\
                +f"Min.  Avg. Val   Loss across Mini-Batch = {self.min_val_loss *1e6 : .1f} e-6; \n"\
                +f"Curr. Avg. (Val-Train) overfit gap      =  {(current_avg_val_loss - current_avg_train_loss) *1e6 : .1f} e-6; " \
                                   +f"= (1/var)*||X-X_r||^2 val-train = {(self.val_multiple_losses_avg['reconstruction_loss'][-1] - self.train_multiple_losses_avg['reconstruction_loss'][-1])*1e6 :.1f} e-6 " \
                                   +f"and {non_reconstruction_loss_str}*||Z_e-Z_q||^2 val-train = {((self.val_multiple_losses_avg['VQ_codebook_loss'][-1] +  self.model.args_VQ['beta'] * self.val_multiple_losses_avg['commitment_loss'][-1]) - (self.train_multiple_losses_avg['VQ_codebook_loss'][-1] +  self.model.args_VQ['beta'] * self.train_multiple_losses_avg['commitment_loss'][-1]))*1e6 :.1f} e-6 \n"\
                +f"\n----------------------------------------------------------------------------------\n"
        
        return intermediate_training_stats
    
    def train(self) -> None:
        #torch.autograd.set_detect_anomaly(True)
        print("Training Started")
        
        # time when the training began
        START_TIME_TRAINING = time.time()
        
        # put the model to the specified device
        self.model = self.model.to(self.device)
        
        # put the loss to the specified device
        self.loss_fn.to(self.device)

        # training and validation minimal avg. losses
        # (avg. loss is the loss averaged in one mini-batch sample)
        self.min_train_loss, self.min_val_loss = np.inf, np.inf
        
        # rememeber training and validation avg. losses
        self.train_loss_avg = []
        self.val_loss_avg = []
        
        # rememeber multiple training and validation avg. losses
        self.usage_of_multiple_terms_loss_function = True
        if self.usage_of_multiple_terms_loss_function:
            self.train_multiple_losses_avg = {}
            self.val_multiple_losses_avg = {}
            self.train_multiple_losses_avg['reconstruction_loss'] = []
            self.train_multiple_losses_avg['commitment_loss'] = []#e_latent_loss = || Z_e - Z_q.detach() || ^ 2
            self.train_multiple_losses_avg['VQ_codebook_loss'] = []#q_latent_loss = || Z_e.detach() - Z_q || ^ 2
            self.val_multiple_losses_avg['reconstruction_loss'] = []
            self.val_multiple_losses_avg['commitment_loss'] = []#e_latent_loss = || Z_e - Z_q.detach() || ^ 2
            self.val_multiple_losses_avg['VQ_codebook_loss'] = []#q_latent_loss = || Z_e.detach() - Z_q || ^ 2
        
        # init the additional metrics for training and validation
        self.train_metrics = {}
        self.val_metrics = {}
        
        # init the perplexity array
        self.train_metrics['perplexity'] = []
        self.val_metrics['perplexity'] = []
        
        # training and validation duration in seconds per epoch
        self.train_duration_per_epoch_seconds = [None]*self.NUM_EPOCHS
        self.val_duration_per_epoch_seconds = [None]*self.NUM_EPOCHS
        
        #init early stopper: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
        early_stopping = EarlyStopper(patience=1, min_delta=0)
        
        # epochs loop
        for epoch in range(self.NUM_EPOCHS):
            
            ###################
            # train the model #
            ###################
            
            # time when the training epoch started
            start_time_epoch = time.time()
            
            # init the avaraged training loss
            self.train_loss_avg.append(0.)
            
            # init the avaraged training multiple losses in training part
            self.train_multiple_losses_avg['reconstruction_loss'].append(0.)
            self.train_multiple_losses_avg['commitment_loss'].append(0.)#e_latent_loss = || Z_e - Z_q.detach() || ^ 2
            self.train_multiple_losses_avg['VQ_codebook_loss'].append(0.)#q_latent_loss = || Z_e.detach() - Z_q || ^ 2
                        
            # init the number of mini-batches covered in the current epoch
            num_batches = 0
            
            # init current training loss
            self.train_loss = 0.0
            
            # set the model to the train state
            self.model.train()
            
            for image_batch, image_ids_batch  in self.loaders['train']:
                #train_data_variance = torch.var(image_batch).item()
                
                #torch.Size([BATCH_SIZE_TRAIN, 3 (RGB), H, W])
                image_batch = image_batch.to(self.device)
                
                # autoencoder reconstruction
                # forward pass: compute predicted outputs by passing inputs to the model
                image_batch_recon = self.model(image_batch)
                
                # reconstruction error: calculate the batch loss
                if len(image_batch_recon) == 5:
                    e_and_q_latent_loss, image_batch_recon_, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = image_batch_recon                                       
                    recon_error = F.mse_loss(image_batch_recon_, image_batch) / self.train_data_variance
                    loss = recon_error + e_and_q_latent_loss
                else:
                    loss = self.loss_fn(image_batch_recon, image_batch)
                    
                ## find the loss and update the model parameters accordingly
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                
                # (backpropagation) backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                
                # one step of the optmizer (using the gradients from backpropagation)
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                
                # sum up the current training loss in the last element of the train_loss_avg
                self.train_loss_avg[-1] += loss.item()
                
                # sum up the current training loss in the last element of array for every loss in training part
                if self.usage_of_multiple_terms_loss_function:
                    self.train_multiple_losses_avg['reconstruction_loss'][-1] += recon_error.item()
                    self.train_multiple_losses_avg['commitment_loss'][-1]     += e_latent_loss #e_latent_loss = || Z_e - Z_q.detach() || ^ 2
                    self.train_multiple_losses_avg['VQ_codebook_loss'][-1]    += q_latent_loss #q_latent_loss = || Z_e.detach() - Z_q || ^ 2
                                    
                # count the number of batches
                num_batches += 1
                
            # calculate the current avg. training loss
            self.train_loss_avg[-1] /= (1.*num_batches)
            
            # calculate the current avg. training losses per term in training part
            if self.usage_of_multiple_terms_loss_function:
                self.train_multiple_losses_avg['reconstruction_loss'][-1] /= (1.*num_batches)
                self.train_multiple_losses_avg['commitment_loss'][-1]     /= (1.*num_batches)
                self.train_multiple_losses_avg['VQ_codebook_loss'][-1]    /= (1.*num_batches)
            
            # calculate the current min. avg. training loss
            self.min_train_loss = np.min([self.min_train_loss, self.train_loss_avg[-1]])
            
            # save additional metric - perplexity = exp(entropy)
            self.train_metrics['perplexity'].append(estimate_codebook_words_exp_entropy)
            
            # calculate training elapsed time in the current epoch
            self.train_duration_per_epoch_seconds[epoch] = int(time.time() - start_time_epoch)
            
            ######################    
            # validate the model #
            ######################
            
            # time when the validation epoch started
            start_time_epoch = time.time()
            
            # init the avaraged validation loss
            self.val_loss_avg.append(0.)
            
            # init the avaraged val. multiple losses in validation part
            self.val_multiple_losses_avg['reconstruction_loss'].append(0.)
            self.val_multiple_losses_avg['commitment_loss'].append(0.)#e_latent_loss = || Z_e - Z_q.detach() || ^ 2
            self.val_multiple_losses_avg['VQ_codebook_loss'].append(0.)#q_latent_loss = || Z_e.detach() - Z_q || ^ 2
                      
            # init the number of mini-batches covered in the current epoch
            num_batches = 0
            
            # init current validation loss
            self.valid_loss = 0.0
            
            # set the model to the evaluation state
            self.model.eval()
            
            for image_batch, image_ids_batch in self.loaders['val']:
                
                #torch.Size([BATCH_SIZE_VAL, 3 (RGB), H, W])
                image_batch = image_batch.to(self.device)
                
                # autoencoder reconstruction 
                # forward pass: compute predicted outputs by passing inputs to the model
                image_batch_recon = self.model(image_batch)
                
                # reconstruction error
                # calculate the batch loss
                if len(image_batch_recon) == 5:
                    e_and_q_latent_loss, image_batch_recon_, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = image_batch_recon                                       
                    recon_error = F.mse_loss(image_batch_recon_, image_batch) / self.train_data_variance
                    loss = recon_error + e_and_q_latent_loss
                else:
                    loss = self.loss_fn(image_batch_recon, image_batch)
                    
                
                # since this is validation of the model there is not (backprogagation and update step size)
                  
                # sum up the current validation loss in the last element of the val_loss_avg
                self.val_loss_avg[-1] += loss.item()
                
                # sum up the current validation loss in the last element of array for every loss in validation part
                if self.usage_of_multiple_terms_loss_function:
                    self.val_multiple_losses_avg['reconstruction_loss'][-1] += recon_error.item()
                    self.val_multiple_losses_avg['commitment_loss'][-1]     += e_latent_loss #e_latent_loss = || Z_e - Z_q.detach() || ^ 2
                    self.val_multiple_losses_avg['VQ_codebook_loss'][-1]    += q_latent_loss #q_latent_loss = || Z_e.detach() - Z_q || ^ 2

                # count the number of batches
                num_batches += 1
            
            # calculate the current avg. validation loss
            self.val_loss_avg[-1] /= (1.*num_batches)
            
            # calculate the current avg. val. losses per term in validation part
            if self.usage_of_multiple_terms_loss_function:
                self.val_multiple_losses_avg['reconstruction_loss'][-1] /= (1.*num_batches)
                self.val_multiple_losses_avg['commitment_loss'][-1]     /= (1.*num_batches)
                self.val_multiple_losses_avg['VQ_codebook_loss'][-1]    /= (1.*num_batches)
                
            # calculate the current min. avg. validation loss
            self.min_val_loss = np.min([self.min_val_loss, self.val_loss_avg[-1]])
            
            # save additional metric - perplexity = exp(entropy)
            self.val_metrics['perplexity'].append(estimate_codebook_words_exp_entropy)
            
            # calculate validation elapsed time in the current epoch
            self.val_duration_per_epoch_seconds[epoch] = int(time.time() - start_time_epoch)
            
            # scheduler stop
            #self.scheduler.step()
            
            # every 10th epoch print intermediate training/validation statistics
            if (epoch+1) % int(0.05 * self.NUM_EPOCHS) == 0:
                
                
                message = self.get_intermediate_training_stats_str(\
                    current_epoch           = epoch,
                    start_time_training     = START_TIME_TRAINING)
                #print(message)
                with open(self.logger_path, 'a') as f:
                    f.write(f"current train      perplexity = 2^( H( PMF of codebook words occurance ) bits ) = { self.train_metrics['perplexity'][-1] :.2f}; perplexity/K = {self.train_metrics['perplexity'][-1] / self.model.args_VQ['K'] * 100 :.2f}%\n")
                    f.write(f"current validation perplexity = 2^( H( PMF of codebook words occurance ) bits ) = { self.val_metrics['perplexity'][-1] :.2f}; perplexity/K = {self.val_metrics['perplexity'][-1] / self.model.args_VQ['K'] * 100 :.2f}%\n")
                    f.write(f"{message}\n")
                
                if (epoch+1)==int(0.2 * self.NUM_EPOCHS):
                    if (self.train_loss_avg[epoch]/self.train_loss_avg[epoch-int(0.05 * self.NUM_EPOCHS)]) > 1.0: # self.train_loss_avg[-1]> 16000*1e-6: #self.train_loss_avg[-1] > 16000*1e-6
                        print(f"The model is not learning, i.e. mini-batch avg. train loss in epoch {epoch} = {self.train_loss_avg[epoch]  *1e6 : .1f} e-6, while in epoch {epoch-10} avg. tr. loss = {self.train_loss_avg[epoch-10]  *1e6 : .1f} e-6, and their ratio = {(self.train_loss_avg[epoch] / self.train_loss_avg[epoch-10])  : .2f} \n")
                        #raise KeyboardInterrupt #^C # keyboard interruption = means the model is not learning
                
                if (epoch+1)==int(0.5 * self.NUM_EPOCHS) or (epoch+1)==self.NUM_EPOCHS:# and self.train_loss_avg[-1] < 16000*1e-6: #self.train_loss_avg[-1] > 16000*1e-6
                    with open(self.logger_path, 'a') as f:
                        k_,d_,m_ = self.model.args_VQ['K'], self.model.args_VQ['D'], self.model.args_VQ['M']
                        f.write(f"The model is learning, for K = {k_}, D= {d_}, M = {m_}\n")
                        f.write(f"{message}\n")
                    #raise KeyboardInterrupt
                if self.PCA_decomp_in_every_epochs:
                    self.plot_codebook_PCA(epoch_id = epoch+1)

                    # for layers_ in [self.model.encoder.sequential_convs, self.model.decoder.sequential_trans_convs]:
                        # for x in layers_:
                        #     try:
                        #         if x._get_name()[:4] == "Conv":
                        #             print(x._get_name(),f"\t {x.weight.detach().cpu().norm().item() :.1f}")
                        #     except:
                        #         print('')
                # if self.train_loss_avg[-1] < 4000*1e-6:
                #     k_,d_,m_ = self.model.args_VQ['K'], self.model.args_VQ['D'], self.model.args_VQ['M']
                #     print(f"The model is learning, for K = {k_}, D= {d_}, M = {m_}")
                #     raise KeyboardInterrupt
            
            # DO NOT DO EARLY STOPPING
            # early stopping
            if False and (epoch+1) >= int(0.9* self.NUM_EPOCHS):
                #if early_stopping(validation_loss = self.val_loss_avg[-1]):
                if 1.05 * self.min_val_loss >= self.val_loss_avg[-1]:
                    print(self.get_intermediate_training_stats_str(current_epoch = epoch, start_time_training = START_TIME_TRAINING))
                    with open(self.logger_path, 'a') as f:
                        f.write(f"\n!!! Early stopping happened at the epochs = {(epoch+1)}. And the current train/val loss message is the following!!! \n")
                        f.write(f"{message}\n")
                    break
                    
        
        with open(self.logger_path, 'a') as f:
            k_,d_,m_ = self.model.args_VQ['K'], self.model.args_VQ['D'], self.model.args_VQ['M']
            f.write(f"The model is learning, for K = {k_}, D= {d_}, M = {m_}\n")
        
        # get current time in the format YYYY_MM_DD_hh_mm_ss
        self.current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())) # 2022_11_19_20_11_26
        #train_loss_avg_path = '/home/novakovm/iris/MILOS/autoencoder_train_loss_avg_' + current_time_str + '.npy'
        
        # create training/validation avg. loss file paths
        self.train_loss_avg_path = self.main_folder_path + self.model_name + '_train_loss_avg_' + self.current_time_str + '.npy'
        self.val_loss_avg_path = self.main_folder_path + self.model_name + '_val_loss_avg_' + self.current_time_str + '.npy'

        # create training/validation avg. loss file paths per loss term
        self.train_multiple_losses_avg_path = {}
        self.val_multiple_losses_avg_path = {}
        for loss_term in ['reconstruction_loss','commitment_loss', 'VQ_codebook_loss']:
            #'reconstruction_loss' = || x - x_recon || ^ 2
            #'commitment_loss'     = e_latent_loss = || Z_e - Z_q.detach() || ^ 2
            #'VQ_codebook_loss'    = q_latent_loss = || Z_e.detach() - Z_q || ^ 2
            self.train_multiple_losses_avg_path[loss_term] = self.main_folder_path  + self.model_name + '_train_multiple_losses_avg_' + loss_term + '_'  + self.current_time_str + '.npy'
            self.val_multiple_losses_avg_path[loss_term] = self.main_folder_path + self.model_name + '_val_multiple_losses_avg_' + loss_term + '_'  + self.current_time_str + '.npy'

        # create training/validation avg. losses as numpy arrays to the above specified file paths
        self.train_loss_avg = np.array(self.train_loss_avg)
        self.val_loss_avg = np.array(self.val_loss_avg)
        np.save(self.train_loss_avg_path,self.train_loss_avg)
        np.save(self.val_loss_avg_path,self.val_loss_avg)
        
        # create training/validation avg. losses per loss term as numpy arrays to the above specified file paths
        for loss_term in ['reconstruction_loss','commitment_loss', 'VQ_codebook_loss']:
                self.train_multiple_losses_avg[loss_term] = np.array(self.train_multiple_losses_avg[loss_term])
                self.val_multiple_losses_avg[loss_term] = np.array(self.val_multiple_losses_avg[loss_term])
                np.save(self.train_multiple_losses_avg_path[loss_term], self.train_multiple_losses_avg[loss_term])
                np.save(self.val_multiple_losses_avg_path[loss_term], self.val_multiple_losses_avg[loss_term])
        
        self.train_metrics['perplexity'] = np.array(self.train_metrics['perplexity'])
        self.val_metrics['perplexity'] = np.array(self.val_metrics['perplexity'])
        
        self.train_metrics_perplexity_path = self.main_folder_path + self.model_name + '_train_perplexity_' + self.current_time_str + '.npy'
        self.val_metrics_perplexity_path = self.main_folder_path + self.model_name + '_val_perplexity_' + self.current_time_str + '.npy'
        
        np.save(self.train_metrics_perplexity_path, self.train_metrics['perplexity'])
        np.save(self.val_metrics_perplexity_path, self.val_metrics['perplexity'])
        
        # save M_ema and N_ema
        # if self.model.VQ.use_EMA:
        #     torch.save(self.model.N_ema, self.main_folder_path \
        #                                 + self.model_name + '_N_ema_' \
        #                                 + self.current_time_str + '.pt')
        #     torch.save(self.model.M_ema, self.main_folder_path \
        #                                 + self.model_name + '_M_ema_' \
        #                                 + self.current_time_str + '.pt')
        
        # Message that the training/validation avg. losses are saved and the corresponding paths
        #print(f"Autoencoder Training Loss Average saved here\n{self.train_loss_avg_path}", end = '\n\n')
        #print(f"Autoencoder Validation Loss Average saved here\n{self.val_loss_avg_path}", end = '\n\n')

        # example of self.model_path:=            
        #/home/novakovm/iris/MILOS
        # /
        #vanilla_autoencoder
        # 2022_12_03_19_39_08
        #.py
        self.model_path = self.main_folder_path + self.model_name + self.current_time_str + '.py'
        
        # saving model trained parameters, so that it could be used as a pretrained model in the future usages
        torch.save(self.model.state_dict(),self.model_path)
        print(f"Current Trained Model saved at = \n {self.model_path}", end = '\n\n')
        
        TOTAL_TRAINING_TIME = int(time.time() - START_TIME_TRAINING)
        m, s = divmod(TOTAL_TRAINING_TIME, 60)
        h, m = divmod(m, 60)
        TOTAL_TRAINING_TIME = f"{h}:{m}:{s} h/m/s"
        
        print(f"Total training time is = {TOTAL_TRAINING_TIME}, end = '\n\n'")
        print("Training Ended", end = '\n--------------------------------------------------------------\n')
    
    def load_model(self, current_time_str, autoencoder_config_params_wrapped_sorted = None) -> None:
        #current time in the format YYYY_MM_DD_hh_mm_ss
        self.current_time_str  = current_time_str
        self.model_path = self.main_folder_path + self.model_name + self.current_time_str + '.py'
        
        # load model architecture in params wrapped and sorted fashion
        #self.autoencoder_config_params_wrapped_sorted = autoencoder_config_params_wrapped_sorted
        
        # create a model (constructor)
        #self.model = Vanilla_Autoencoder_v02(autoencoder_config_params_wrapped_sorted=self.autoencoder_config_params_wrapped_sorted)
        #self.model = model
        
        # load the model state from the model path
        self.model.load_state_dict(torch.load(self.model_path))
    
        # N_ema = torch.load(self.main_folder_path  \
        #                     + self.model_name + '_N_ema_' \
        #                     + self.current_time_str + '.pt')
        # M_ema = torch.load(self.main_folder_path  \
        #                     + self.model_name + '_M_ema_' \
        #                     + self.current_time_str + '.pt')
        # move model to device
        self.model = self.model.to(device=self.device)
        
        # put loaded model in the evaulation mode
        self.model.eval()
    
    def test(self) -> None:
        ######################    
        # testing the model #
        ######################
        print("Testing Started")

        # init test loss array per mini-batch as an empty array
        #self.test_loss = []

        # test_samples_loss is pandas DataFrame that has two columns
        # first column name is the test_image_id that is the id of the test image (int type)
        # second column name is the test_image_rec_loss that is the reconstruction loss of the test image with the id equal to test_image_id (float type)
        self.test_samples_loss = {} # test_image_tensor: test_loss
        self.test_samples_loss['test_image_id'] = []
        
        #self.test_samples_loss['test_image_rec_loss'] = []
        self.test_samples_loss['total_loss'] = []
        self.test_samples_loss['reconstruction_loss'] = [] #reconstruction_loss = || X - X_rec || ^ 2 
        self.test_samples_loss['commitment_loss'] = []  #e_latent_loss = || Z_e - Z_q.detach() || ^ 2
        self.test_samples_loss['VQ_codebook_loss'] = [] #q_latent_loss = || Z_e.detach() - Z_q || ^ 2

        
        self.test_metrics = {}
        self.test_metrics['perplexity'] = []

        # put the model to the specified device
        self.model = self.model.to(self.device)

        # put the loss to the specified device
        self.loss_fn.to(self.device)

        # put loaded model in the evaulation mode
        self.model.eval()

        for image_batch, image_id_batch in self.loaders['test']:
            if self.loaders['test'].batch_size != 1:
                assert(self.loaders['test'].batch_size == 1, f"Mini-batch size of the test set should be 1, because of visualization and plotting later on in the code.")
            
            # remember the test_image_id (i.e. id of the test image)
            self.test_samples_loss['test_image_id'].append(image_id_batch.item())
                
            with torch.no_grad():
                # move the image tensor to the device
                image_batch = image_batch.to(self.device)
                
                # autoencoder reconstruction (forward pass)
                image_batch_recon = self.model(image_batch)

                # reconstruction error (loss calculation)
                if len(image_batch_recon) == 5:
                    e_and_q_latent_loss, image_batch_recon_, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = image_batch_recon                                       
                    recon_error = F.mse_loss(image_batch_recon_, image_batch)/ self.train_data_variance
                    loss = recon_error + e_and_q_latent_loss
                    
                    self.test_samples_loss['total_loss'].append(loss.item()) # reconstruction_loss + q_latent_loss + beta * e_latent_loss
                    self.test_samples_loss['reconstruction_loss'].append(recon_error.item()) #reconstruction_loss = (1/var)|| X - X_rec || ^ 2 
                    self.test_samples_loss['commitment_loss'].append(e_latent_loss)     #e_latent_loss = || Z_e - Z_q.detach() || ^ 2
                    self.test_samples_loss['VQ_codebook_loss'].append(q_latent_loss)    #q_latent_loss = || Z_e.detach() - Z_q || ^ 2
                    
                    self.test_metrics['perplexity'].append(estimate_codebook_words_exp_entropy)
                else:
                    loss = self.loss_fn(image_batch_recon, image_batch)
                
                # remember the test_image_id's reconstruction loss (in a different array used for complex plotting)
                #self.test_samples_loss['test_image_rec_loss'].append(loss.item())
                
                # remember the test_image_id's reconstruction loss (in a different array used for simple plotting)
                #self.test_loss.append(loss.item())

        # cast it to np.array type 
        #self.test_loss = np.array(self.test_loss)
        
        # cast to np array
        self.test_samples_loss['test_image_id'] = np.array(self.test_samples_loss['test_image_id'])

        self.test_samples_loss['total_loss'] = np.array(self.test_samples_loss['total_loss'])
        self.test_samples_loss['reconstruction_loss'] = np.array(self.test_samples_loss['reconstruction_loss'])
        self.test_samples_loss['commitment_loss'] = np.array(self.test_samples_loss['commitment_loss'])
        self.test_samples_loss['VQ_codebook_loss'] = np.array(self.test_samples_loss['VQ_codebook_loss'])

        self.test_metrics['perplexity'] = np.array(self.test_metrics['perplexity'])
        #self.test_metrics_perplexity_path = self.main_folder_path + '/' + self.model_name + '_test_perplexity_' + self.current_time_str + '.npy'
        #np.save(self.test_metrics_perplexity_path, self.test_metrics['perplexity'])
        
        print(f"Average test total loss = {self.test_samples_loss['total_loss'].mean() * 1e6 : .0f} e-6")
        print(f"Average test total reconstruction loss = {self.test_samples_loss['reconstruction_loss'].mean() * 1e6 : .0f} e-6 ({ self.test_samples_loss['reconstruction_loss'].mean()/self.test_samples_loss['total_loss'].mean() *100:.0f} %)")
        print(f"Average test total commitment loss = {self.model.VQ.beta * self.test_samples_loss['commitment_loss'].mean() * 1e6 : .0f} e-6 ({self.model.VQ.beta *  self.test_samples_loss['commitment_loss'].mean()/self.test_samples_loss['total_loss'].mean() *100:.0f} %)")
        print(f"Average test total VQ loss = {self.test_samples_loss['VQ_codebook_loss'].mean() * 1e6 : .0f} e-6 ({ self.test_samples_loss['VQ_codebook_loss'].mean()/self.test_samples_loss['total_loss'].mean() *100:.0f} %)")
        
        
        print("Testing Ended")

    def plot(self, train_val_plot = True, test_plot = True) -> None:
        # Plot Training and Validation Average Loss per Epoch
        if train_val_plot:
            # SEMILOG-Y SCALE for both validation and training loss
            plt.figure()
            legend_labels = []
                        
            # Semilogy plot for avg. training losses
            plt.semilogy(self.train_loss_avg, '-k') # black solid line
            legend_labels.append('Training Loss')
            
            # Semilogy plot for avg. validation losses
            plt.semilogy(self.val_loss_avg, '-m') # magenta solid line
            legend_labels.append('Validation Loss')

            plt.title(f'Train (Min. = {self.train_loss_avg.min() *1e3: .2f} e-3) & '\
                    + f'Validation (Min. = {self.val_loss_avg.min() *1e3: .2f} e-3) \n '\
                    + f'Loss Averaged across Mini-Batch per epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Mini-Batch Avg. Train & Validation Loss')
            plt.legend(legend_labels)
            plt.grid()
            plt.savefig(self.main_folder_path + 'semilog_train_val_loss_per_epoch.png')
            plt.close()
            
            TOTAL_EPOCH_NUMBER = len(self.train_loss_avg)
            # SEMILOG-Y SCALE for individual Loss terms applied to training loss
            DROP_FIRST_N_EPOCHS = int(0.05 * TOTAL_EPOCH_NUMBER)
            
            X_AXIS = np.arange(DROP_FIRST_N_EPOCHS, TOTAL_EPOCH_NUMBER)
            plt.figure(figsize=(45,15))
            legend_labels = []
            
            # Total Training Loss (sum of all terms)
            plt.semilogy(X_AXIS, self.train_loss_avg[DROP_FIRST_N_EPOCHS:], '-k') # black solid line
            legend_labels.append('Training Loss')
            
            # Total Validation Loss (sum of all terms)
            # plt.semilogy(X_AXIS, self.val_loss_avg[DROP_FIRST_N_EPOCHS:], '--m') # magenta solid line
            # legend_labels.append('Validation Loss')
                        
            # semilogy also the individual loss terms for both training and validaiton sets            
            if self.usage_of_multiple_terms_loss_function:
                # fmt = '[marker][line][color]' look here for full table of descriptors for plots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
                # Training Loss Terms:
                
                # Reconstruction Loss Term
                marker, line, color = '', '-', 'r'
                fmt = marker + line + color
                plt.semilogy(X_AXIS, self.train_multiple_losses_avg['reconstruction_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                legend_labels.append('Training Reconstruction Loss := || X - X_recon || ^ 2')
                
                # beta * Commitment Loss Term
                marker, line, color = '', '-', 'g'
                fmt = marker + line + color
                plt.semilogy(X_AXIS, self.model.args_VQ['beta'] * self.train_multiple_losses_avg['commitment_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                legend_labels.append('Training Commitment Loss := ' + r'$\beta$' + ' * || Z_e - Z_q.detach() || ^ 2')
                
                # VQ_codebook_loss
                marker, line, color = '', '-', 'b'
                fmt = marker + line + color
                plt.semilogy(X_AXIS, self.train_multiple_losses_avg['VQ_codebook_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                legend_labels.append('Training VQ Loss := || Z_e.detach() - Z_q || ^ 2')
                
                # # Validation Loss Terms:
                
                # # Reconstruction Loss Term
                # marker, line, color = 'o', '--', 'r'
                # fmt = marker + line + color
                # plt.semilogy(X_AXIS, self.val_multiple_losses_avg['reconstruction_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                # legend_labels.append('Validation Reconstruction Loss := || X - X_recon || ^ 2')
                
                # # beta * Commitment Loss Term
                # marker, line, color = 'o', '--', 'g'
                # fmt = marker + line + color
                # plt.semilogy(X_AXIS, self.model.args_VQ['beta'] * self.val_multiple_losses_avg['commitment_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                # legend_labels.append('Validation Commitment Loss := ' + r'$\beta$' + ' * || Z_e - Z_q.detach() || ^ 2')
                
                # # VQ_codebook_loss
                # marker, line, color = 'o', '--', 'b'
                # fmt = marker + line + color
                # plt.semilogy(X_AXIS, self.val_multiple_losses_avg['VQ_codebook_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                # legend_labels.append('Validation VQ Loss := || Z_e.detach() - Z_q || ^ 2')
            
            plt.title(f'Training Loss Averaged across Mini-Batch per epoch (Min. = {self.train_loss_avg.min() *1e3: .2f} e-3)')
            plt.xlabel('Epochs')
            plt.ylabel('Mini-Batch Avg. Train Loss')
            plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.savefig(self.main_folder_path + 'semilog_train_per_loss_terms_per_epoch.png')
            plt.close()
            
            
            # SEMILOG-Y SCALE for individual Loss terms applied to validation loss
            TOTAL_EPOCH_NUMBER = len(self.train_loss_avg)
            DROP_FIRST_N_EPOCHS = int(0.05*TOTAL_EPOCH_NUMBER)
            X_AXIS = np.arange(DROP_FIRST_N_EPOCHS, TOTAL_EPOCH_NUMBER)
            plt.figure(figsize=(45,15))
            legend_labels = []
            
            # # Total Training Loss (sum of all terms)
            # plt.semilogy(X_AXIS, self.train_loss_avg[DROP_FIRST_N_EPOCHS:], '-k') # black solid line
            # legend_labels.append('Training Loss')
            
            # Total Validation Loss (sum of all terms)
            plt.semilogy(X_AXIS, self.val_loss_avg[DROP_FIRST_N_EPOCHS:], '--m') # magenta solid line
            legend_labels.append('Validation Loss')
                        
            # semilogy also the individual loss terms for both training and validaiton sets            
            if self.usage_of_multiple_terms_loss_function:
                # fmt = '[marker][line][color]' look here for full table of descriptors for plots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
                # Training Loss Terms:
                
                # # Reconstruction Loss Term
                # marker, line, color = '', '-', 'r'
                # fmt = marker + line + color
                # plt.semilogy(X_AXIS, self.train_multiple_losses_avg['reconstruction_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                # legend_labels.append('Training Reconstruction Loss := || X - X_recon || ^ 2')
                
                # # beta * Commitment Loss Term
                # marker, line, color = '', '-', 'g'
                # fmt = marker + line + color
                # plt.semilogy(X_AXIS, self.model.args_VQ['beta'] * self.train_multiple_losses_avg['commitment_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                # legend_labels.append('Training Commitment Loss := ' + r'$\beta$' + ' * || Z_e - Z_q.detach() || ^ 2')
                
                # # VQ_codebook_loss
                # marker, line, color = '', '-', 'b'
                # fmt = marker + line + color
                # plt.semilogy(X_AXIS, self.train_multiple_losses_avg['VQ_codebook_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                # legend_labels.append('Training VQ Loss := || Z_e.detach() - Z_q || ^ 2')
                
                # Validation Loss Terms:
                
                # Reconstruction Loss Term
                marker, line, color = '', '-', 'r'
                fmt = marker + line + color
                plt.semilogy(X_AXIS, self.val_multiple_losses_avg['reconstruction_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                legend_labels.append('Validation Reconstruction Loss := || X - X_recon || ^ 2')
                
                # beta * Commitment Loss Term
                marker, line, color = '', '-', 'g'
                fmt = marker + line + color
                plt.semilogy(X_AXIS, self.model.args_VQ['beta'] * self.val_multiple_losses_avg['commitment_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                legend_labels.append('Validation Commitment Loss := ' + r'$\beta$' + ' * || Z_e - Z_q.detach() || ^ 2')
                
                # VQ_codebook_loss
                marker, line, color = '', '-', 'b'
                fmt = marker + line + color
                plt.semilogy(X_AXIS, self.val_multiple_losses_avg['VQ_codebook_loss'][DROP_FIRST_N_EPOCHS:], fmt)
                legend_labels.append('Validation VQ Loss := || Z_e.detach() - Z_q || ^ 2')
            
            plt.title(f'Validation Loss Averaged across Mini-Batch per epoch (Min. = {self.val_loss_avg.min() *1e3: .2f} e-3)')
            plt.xlabel('Epochs')
            plt.ylabel('Mini-Batch Avg. Validation Loss')
            plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.savefig(self.main_folder_path + 'semilog_val_per_loss_terms_per_epoch.png')
            plt.close()
            
        if test_plot:
            # Plot Test Loss for every sample in the Test set
            fig = plt.figure()
            ax = plt.gca()
            x_scatter = self.test_samples_loss['test_image_id']
            y_scatter = self.test_samples_loss['total_loss']  #self.test_samples_loss['test_image_rec_loss']
            colors = "blue"#[5 for img_id in x_scatter]
            area = 20 #(30 * np.random.rand(N))**2  # 0 to 15 point radii
            plt.scatter(x_scatter, y_scatter, s=area, c=colors, alpha=0.4)
            ax.set_yscale('log')
            plt.title(f'Test Loss per sample in the Test set \n'+
                        f'(Avg. = {y_scatter.mean()*1e3 : .2f} e-3)')
            plt.grid()
            plt.xlabel('Test sample ids')
            plt.ylabel('Testing Loss')
            plt.savefig(self.main_folder_path + 'testing_loss_per_image_in_minibatch.png')
            plt.close()
    
    def plot_perlexity(self):
        
        # plot perplexity = 2^(estimated codewords entropy in bits)
        plt.figure(figsize=[6.4, 6.8] )
        plt.plot(self.train_metrics['perplexity'], 'k', label = 'train perplexity')
        plt.plot(self.val_metrics['perplexity'], 'm', label = 'validation perplexity')
        y = self.model.args_VQ['K']
        max_train_perplexity, max_val_perplexity = np.max(self.train_metrics['perplexity']), np.max(self.val_metrics['perplexity'])
        last_train_perplexity, last_val_perplexity = self.train_metrics['perplexity'][-1], self.val_metrics['perplexity'][-1]
        
        plt.axhline(y=y, color='r', linestyle='-', label = f'Ground Truth K = {y}')
        plt.title(f"Training and validation perplexity (2 ^ (estimated codewords entropy))\n"\
                    + f"max. train/val perplexity = { max_train_perplexity :.1f} / { max_val_perplexity :.1f}\n"\
                    + f"last train/val perplexity = { last_train_perplexity :.1f} / { last_val_perplexity :.1f}")
        plt.grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.savefig(self.main_folder_path + 'train_val_Perplexity.png')
        plt.close()
        
        # plot estimated codewords entropy in bits
        plt.figure(figsize=[6.4, 5.8] )
        plt.plot(np.log2(self.train_metrics['perplexity']), 'k', label = 'train entropy estimation [bits]')
        plt.plot(np.log2(self.val_metrics['perplexity']), 'm', label = 'validation entropy estimation [bits]')
        y = np.log2(self.model.args_VQ['K'])
        max_train_entropy, max_val_entropy = np.log2(max_train_perplexity), np.log2(max_val_perplexity)
        last_train_entropy, last_val_entropy = np.log2(last_train_perplexity), np.log2(last_val_perplexity)
        
        plt.axhline(y=y, color='r', linestyle='-', label = f'Ground Truth log2(K) = {y} [bits]')

        
        plt.title(f"Training and validation estimated codewords entropy H(E)\n"\
            + f"max. train/val entropy = { max_train_entropy :.1f} / { max_val_entropy :.1f}\n"\
            + f"last train/val entropy = { last_train_entropy :.1f} / { last_val_entropy :.1f}")
        plt.grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Estimated codewords entropy H(E) in bits')
        plt.legend()
        plt.savefig(self.main_folder_path + 'train_val_estimated_codewords_entropy.png')
        plt.close()
        
            
    def scatter_plot_test_images_with_specific_classes(self, shape_features_of_interest) -> None:
        # scatter plot test images with specific classes [Plot Test Loss for every sample in the Test set]
           
        # # one example of encoding a determionistic image generation
        # if [
        #     0,      #code for -> 'shape_thickness':  -1
        #     1,      #code for -> 'shape_name': 'Parallelogram'
        #     1, 1,   #code for -> 'shape_center_x': 54 # = 0.75 * W + coor_begin_x
        #     1, 0,   #code for -> 'shape_center_y': 22 # = 0.5 * H + coor_begin_y
        #     0, 0,   #code for -> 'shape_color': (255, 0, 0)
        #     0, 1,   #code for -> 'a': 12 # = 0.125 * W
        #     1, 1,   #code for -> 'b': 16 # = 0.25 * H
        #     0, 1    #code for -> 'alpha':60
        #     ] == image_binary_code:
        #     ## image_id = 11806 = = '0b10111000011110' = reverse('01111000011101') = reverse([0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1]) = reverse(image_binary_code)
        #     print(image_binary_code)
        #     print(shape_specific_stats)
        
        x_scatter = self.test_samples_loss['test_image_id']
        y_scatter = self.test_samples_loss['total_loss']
        
        # all names have to be unique
        all_shape_features           = ['FILL_NOFILL',
                                        'SHAPE_TYPE_SPACE',
                                        'X_CENTER_SPACE',
                                        'Y_CENTER_SPACE',
                                        'COLOR_LIST',
                                        'a_CENTER_SPACE',
                                        'b_CENTER_SPACE',
                                        'alpha_CENTER_SPACE'
                                        ]

        # location of the yaml config file
        milos_config_path = '/home/novakovm/iris/MILOS/milos_config.yaml'
        
        # Open the file and load the file
        with open(milos_config_path) as f:
            data = yaml.load(f, Loader=SafeLoader)

        # shape_features that we are going to cluster/classify based on test image id
        shape_features = OrderedDict()
        
        # init bit counter to least significant bit (LSB) value, i.e. zero
        bit_counter = 0
        
        # for every shape feature that is in the all_shape_features
        for shape_feature in all_shape_features:
            # idea of this for loop is shown in the following example:
            # e.g.
            # from 6-th bit the specification of 'shape_color' starts and it has lenght 2 bit (i.e. bits on the positions 6 and 7)
            # shape_features['shape_color'] = [['blue', 'green', 'red', 'white'], [6,7]]
            
            # get the list of all possible value shape_feature could take
            shape_feature_all_possible_values = data[shape_feature]
            
            # calculate the minimum number of bits to store shape_feature_all_possible_values
            shape_feature_all_possible_values_nb_of_bits_req = int(np.log2(len(shape_feature_all_possible_values)))
            
            # calculate the minimun bit position for that particular shape_feature
            min_bit = bit_counter
            
            # calculate the maximum bit position for that particular shape_feature
            max_bit = bit_counter + shape_feature_all_possible_values_nb_of_bits_req
            
            # all names have to be unique
            shape_features[shape_feature] = [shape_feature_all_possible_values, 
                                            list(np.arange(min_bit,max_bit))]
            
            # move bit counter to the next max bit
            bit_counter = max_bit

        # from 0-th bit the specification of 'shape_thickness' starts and it has lenght 1 bit (i.e. bit on the position 0)
        if shape_features['FILL_NOFILL'][0][0] == -1:
            shape_features['FILL_NOFILL'][0][0] = 'Fill'    # instead of -1
            shape_features['FILL_NOFILL'][0][1] = 'Hollow'  # instead of +1,+2,.. (the thickness of the line)
            
        # keep just only shape_features that are inside shape_features_of_interest
        shape_features_subset = OrderedDict()
        for shape_feature in shape_features:
            if shape_feature in shape_features_of_interest:
                shape_features_subset[shape_feature] = shape_features[shape_feature]
        
        # overwrite the all features with just the subset of features (determined by shape_features_of_interest)
        shape_features = shape_features_subset  
        
        # init different classes of shape_feature (i.e. code words), as well as their values (i.e. codes)
        classes, class_values= [], []
        
        for shape_features_tuples in itertools.product(*([shape_features[k][0] for k in shape_features][::-1])):
            # itertools.product gives us all combinations of shape_features in the reverse order with [::-1] slicer (because we want to respect the bit order)
            # shape_features_tuples is a tuple of different shape features
            # e.g.
            # for selecting 'FILL_NOFILL', 'SHAPE_TYPE_SPACE', and, 'b_CENTER_SPACE',
            # we can get  classes[-1] to be equal to "0.25-Parallelogram-Fill"
            # and that is our code word for a particular unique combination of shape features
            
            classes.append('-'.join([str(shape_features_tuple) for shape_features_tuple in shape_features_tuples]))
            
            # init class value (i.e. code value)
            class_value = 0
            
            # for every particular code word
            for shape_features_tuple in shape_features_tuples:  
                # for every feature shape
                for shape_feature in shape_features:
                    # check if the particular code word is in the list of feature shape names
                    if shape_features_tuple in shape_features[shape_feature][0]:
                        # if it is, we calculate the code value for that particular code word
                        # that is in the feature shape names;
                        # the code value is calculated using simple bin. to dec. arithmetic
                        
                        # get the position of that shape_features_tuple in a list defined by shape_feature
                        shape_features_tuple_position = shape_features[shape_feature][0].index(shape_features_tuple)
                        
                        # bit-shift the position to the left to get the code value for that particular shape_feature;
                        # we shift the position by amount equal to the LSB for that particular shape_feature;
                        class_value +=   shape_features_tuple_position* 2 ** min(shape_features[shape_feature][1])
                        
                        # print out code word (i.e. classes[-1]) and code value (i.e. class_value)  
                        #print(f"class_value = {class_value} for class = {classes[-1]}")
            
            # save the current class value (i.e. accumulated code value) for that particular classes[-1] (i.e. code word)
            class_values.append(class_value)

        # make class values unique (because there are a lot of unwanted duplicates)
        class_values = list(np.unique(class_values))
        
        # allocate memory for different values associated with different test image ids;
        # values will be used to label every test image id (as it is generated from some deterministic process)
        values = [1] * len(x_scatter)
        
        # for every test image id
        for i, x_scatter_val in enumerate(x_scatter):
            # for every class value (i.e. code word)
            for class_value in class_values:
                # check if that test image id has the same bits (at the right positions) as class value (i.e. code word)
                if x_scatter_val & class_value == class_value:
                    # if it has, that means that value[i] is the index position of class values array based on the associated code word
                    values[i] = class_values.index(class_value)
        
        # get number of all possible combinations
        num_of_all_possible_combinations = len(classes)
        
        # create a color map for plotting purposes
        cmap = plt.cm.get_cmap('jet', num_of_all_possible_combinations)
        
        # create colors array based on the color map and total number of available classes
        # these colors are used for horizontal lines
        colors = [ cmap(x) for x in np.arange(num_of_all_possible_combinations) ]
        
        # set figure size and get ax
        fig = plt.figure(figsize=(16,16))
        ax = plt.gca()
        
        # scatter plot settings
        area = 50
        linewidths = 1.5
        alpha=1.0
        edgecolors = 'black'
        
        # scatter plot with classification of test image ids into the buckets (defined by values list)
        scatter = plt.scatter(x_scatter, 
                              y_scatter,
                              c=values,
                              cmap=cmap,
                              vmin=np.min(values), 
                              vmax=np.max(values),
                              alpha=alpha,
                              s=area,
                              linewidths=linewidths,
                              edgecolors = edgecolors)
        
        # define the plot title when number of shape feature of interest is equal to exactly 1 and if we have more then 1
        if len(shape_features_of_interest) == 1:
            plot_title = f"{shape_features_of_interest[0]}"
        else:
            plot_title = f"{'-'.join(shape_features_of_interest[::-1])}"
        
        
        # scatter plot legend
        plt.legend(handles=scatter.legend_elements(num=None)[0], 
                   labels=classes, 
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.05),
                   fancybox=True,
                   shadow=True,
                   ncol=4,
                   title = f"Possible buckets for generated test images in the format = \n{plot_title}")
        
        # drawing of median and mean horizontal lines for every bucket
        for index_ in np.arange(num_of_all_possible_combinations):
            # non-dotted horizontal line defines the median of the points
            plt.axhline(y = np.median(y_scatter[np.where(values == index_)[0]]),
                        color=colors[index_],
                        linestyle='-',
                        linewidth = 5)
            
            # dotted horizontal line defines the mean of the points
            plt.axhline(y = np.mean(y_scatter[np.where(values == index_)[0]]),
                        color=colors[index_],
                        linestyle='--',
                        linewidth = 5)

        # set the y axis to the log scale (to have semilogy effect)
        ax.set_yscale('log')
        
        # set the plot title
        plt.title(  f'Test Loss per sample in the Test set \n'+
                    f'(Min. = {y_scatter.min()*1e9 : .2f} e-9, '+
                    f'Avg. = {y_scatter.mean()*1e6 : .2f} e-6, '+
                    f'Max. = {y_scatter.max()*1e3 : .2f} e-3)\n'+
                    f'Horizontal lines represent the median (full line) and mean (dotted line) values per class')
        plt.grid()
        plt.xlabel('Test sample ids')
        plt.ylabel('Testing Loss')
        plt.savefig(self.main_folder_path + f"Test_Loss_across_{plot_title}.png")
        plt.close()
            
    def get_worst_test_samples(self, TOP_WORST_RECONSTRUCTED_TEST_IMAGES) -> None:
        # Visualization of top worst reconstructed test images (i.e. where autoencoder fails) 
        self.df_worst_test_samples_loss = pd.DataFrame(self.test_samples_loss)
        self.df_worst_test_samples_loss = self.df_worst_test_samples_loss.sort_values('total_loss',ascending=False)\
                                                            .reset_index(drop=True)
        #pick top- TOP_WORST_RECONSTRUCTED_TEST_IMAGES worst reconstructed images
        self.df_worst_reconstructed_test_images = self.df_worst_test_samples_loss.head(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)
        #print(f"pick top {TOP_WORST_RECONSTRUCTED_TEST_IMAGES} worst reconstructed images\n", self.df_worst_reconstructed_test_images.to_string())


        self.worst_top_images, self.worst_imgs_ids , self.worst_imgs_losses = [], [], []
        for worst_reconstructed_test_image_id, worst_reconstructed_test_image_loss in zip(self.df_worst_reconstructed_test_images['test_image_id'], self.df_worst_reconstructed_test_images['total_loss']):
            # find the test image index when you have test image id in the test_data.image_ids tha
            worst_reconstructed_test_image_id_index = np.where(self.loaders['test'].dataset.image_ids == worst_reconstructed_test_image_id)[0][0]
            
            # get the actual image as well as the image_id
            image, image_id = self.loaders['test'].dataset[worst_reconstructed_test_image_id_index]
            
            # save the test image (tensor)
            self.worst_top_images.append(image)
            
            # save the test image id
            self.worst_imgs_ids.append(image_id)
            
            # save the test image reconstruction error (i.e. loss value)
            self.worst_imgs_losses.append(worst_reconstructed_test_image_loss)

        # saved top_images are list of tensor, so cast to a tensor with torch.stack() function
        self.worst_top_images = torch.stack(self.worst_top_images) #torch.Size(TOP_WORST_RECONSTRUCTED_TEST_IMAGES, C, H, W)
    
    def get_best_test_samples(self, TOP_BEST_RECONSTRUCTED_TEST_IMAGES) -> None:
        # Visualization of top best reconstructed test images (i.e. where autoencoder succeeds) 
        self.df_best_test_samples_loss = pd.DataFrame(self.test_samples_loss)
        self.df_best_test_samples_loss = self.df_best_test_samples_loss.sort_values('total_loss',ascending=True)\
                                                            .reset_index(drop=True)
        #pick top- TOP_BEST_RECONSTRUCTED_TEST_IMAGES best reconstructed images
        self.df_best_reconstructed_test_images = self.df_best_test_samples_loss.head(TOP_BEST_RECONSTRUCTED_TEST_IMAGES)
        #print(f"pick top {TOP_BEST_RECONSTRUCTED_TEST_IMAGES} best reconstructed images\n", self.df_best_reconstructed_test_images.to_string())


        self.best_top_images, self.best_imgs_ids , self.best_imgs_losses = [], [], []
        for best_reconstructed_test_image_id, best_reconstructed_test_image_loss in zip(self.df_best_reconstructed_test_images['test_image_id'], self.df_best_reconstructed_test_images['total_loss']):
            # find the test image index when you have test image id in the test_data.image_ids tha
            best_reconstructed_test_image_id_index = np.where(self.loaders['test'].dataset.image_ids == best_reconstructed_test_image_id)[0][0]
            
            # get the actual image as well as the image_id
            image, image_id = self.loaders['test'].dataset[best_reconstructed_test_image_id_index]
            
            # save the test image (tensor)
            self.best_top_images.append(image)
            
            # save the test image id
            self.best_imgs_ids.append(image_id)
            
            # save the test image reconstruction error (i.e. loss value)
            self.best_imgs_losses.append(best_reconstructed_test_image_loss)

        # saved top_images are list of tensor, so cast to a tensor with torch.stack() function
        self.best_top_images = torch.stack(self.best_top_images) #torch.Size(TOP_WORST_RECONSTRUCTED_TEST_IMAGES, C, H, W)
    
    def visualize_model_as_graph_image(self) -> None:   
        # when running on VSCode run the below command
        # svg format on vscode does not give desired result
        #graphviz.set_jupyter_format('png')
        
        # draw_graph arguments
        B = self.loaders['train'].batch_size
        C, H, W = self.model.C_in, self.model.H_in, self.model.W_in
        filename = f"visualize_model_as_graph_image" # + ".png"

        # taken from https://github.com/mert-kurttutan/torchview
        vq_vae_implemented_model_graph = draw_graph(model = self.model, 
                                                    input_size= (B,C,H,W), 
                                                    graph_name = self.model_name,
                                                    expand_nested=True,
                                                    hide_module_functions = False,
                                                    hide_inner_tensors = False,
                                                    roll = True,
                                                    save_graph = True,
                                                    filename = filename,
                                                    directory = self.main_folder_path)
        # visualize graph with .visual_graph() if in Jupyter Notebook
        # vq_vae_implemented_model_graph.visual_graph
    def plot_codebook_PCA(self, epoch_id = None):
        if not os.path.exists(self.main_folder_path + 'PCA_2D/'):
            os.mkdir(self.main_folder_path + 'PCA_2D/')
        if not os.path.exists(self.main_folder_path + 'PCA_3D/'):
            os.mkdir(self.main_folder_path + 'PCA_3D/')
        if not os.path.exists(self.main_folder_path + 'Cummulative_Explained_Variance_PCA/'):
            os.mkdir(self.main_folder_path + 'Cummulative_Explained_Variance_PCA/')
        
        frames_per_second = 1#10#5#10#1
        format_list = ['mp4', 'gif']
        if epoch_id != None:
            self.epoch_ids_PCA.append(epoch_id)
        
        NUMBER_OF_PRINCIPAL_COMPONENTS = [2,3]
        
        digit_size = len(str(self.NUM_EPOCHS))
        
        for n_components in NUMBER_OF_PRINCIPAL_COMPONENTS:
            E = self.model.VQ.E.weight.data.detach().cpu() # dim = K x D matrix
            # PCA
            pca = PCA(n_components=n_components)
            E_pca = pca.fit(E).transform(E) # dim = K x 2

            # Percentage of variance explained for each components
            # The explained_variance_ratio_ attribute of the PCA() class returns a one-dimensional numpy array which contains the values of the percentage of variance explained by each of the selected components.#https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
            #print("Explained variance ratio in % for the first two: "+ str(pca.explained_variance_ratio_ * 100))
            compoonents_explained_variance_ratio = sum([pca.explained_variance_ratio_[i] for i in range(n_components)])
            
            #plt.figure()
            #colors = ["navy", "turquoise", "darkorange"]
            
            # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            target_name = ["$" +str(x+1)+ "$" for x in range(self.model.VQ.K)]
            
            if n_components == 2:
                plt.figure()
                lw = 1 #linewidth
                for i in range(self.model.VQ.K):
                    plt.scatter(E_pca[i,0], E_pca[i,1], color = 'k', lw=lw, marker = target_name[i])
                
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.grid()
            
                if epoch_id == None:
                    plt.title(f"PCA of the codebook E\n({compoonents_explained_variance_ratio*100 :.1f}% of the data variablility captured by two PCs)")
                    plt.savefig(self.main_folder_path + f"PCA_2D/9999_E_PCA_1st_2nd_PCs.png")
                    
                    if self.PCA_decomp_in_every_epochs:
                        for format_ in format_list:
                            with imageio.get_writer(self.main_folder_path + f"PCA_2PCs_{format_}.{format_}", mode='I', fps = frames_per_second) as writer:
                                for epoch_id_PCA in self.epoch_ids_PCA:
                                    filename = self.main_folder_path + f"PCA_2D/{ str(epoch_id_PCA).zfill(digit_size)}_E_PCA_1st_2nd_PCs.png"
                                    image = imageio.imread(filename)
                                    writer.append_data(image)
                                writer.close()
                                        
                else:
                    plt.title(f"(epoch = {str(epoch_id)}) PCA of the codebook E\n({compoonents_explained_variance_ratio*100 :.1f}% of the data variablility captured by two PCs)")
                    plt.savefig(self.main_folder_path + f"PCA_2D/{ str(epoch_id).zfill(digit_size) }_E_PCA_1st_2nd_PCs.png")
                plt.close()
            
            if n_components == 3:
                
                plt.style.use('default')

                # Prepare 3D graph
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                
                # Plot 3D plot
                lw = 1 #linewidth
                for i in range(self.model.VQ.K):
                    ax.scatter3D(E_pca[i,0], E_pca[i,1], E_pca[i,2], color = 'k', lw=lw, marker = target_name[i])

                # Plot x, y, z even ticks
                # ticks = np.linspace(-6, 6, num=10)
                # ax.set_xticks(ticks)
                # ax.set_yticks(ticks)
                # ax.set_zticks(ticks)
                
                # Plot x, y, z labels
                ax.set_xlabel('PC1')#, rotation=150)
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')#, rotation=60)
                
                ax.grid(True)
            
                if epoch_id == None:
                    plt.title(f"PCA of the codebook E\n({compoonents_explained_variance_ratio*100 :.1f}% of the data variablility captured by three PCs)")
                    plt.savefig(self.main_folder_path + f"PCA_3D/9999_E_PCA_1st_2nd_3rd_PCs.png")
                    
                    if self.PCA_decomp_in_every_epochs:
                        for format_ in format_list:
                            with imageio.get_writer(self.main_folder_path + f"PCA_3PCs_{format_}.{format_}", mode='I',fps = frames_per_second) as writer:
                                for epoch_id_PCA in self.epoch_ids_PCA:
                                    filename = self.main_folder_path + f"PCA_3D/{ str(epoch_id_PCA).zfill(digit_size)}_E_PCA_1st_2nd_3rd_PCs.png"
                                    image = imageio.imread(filename)
                                    writer.append_data(image)
                                writer.close()
                    
                else:
                    plt.title(f"(epoch = {str(epoch_id)}) PCA of the codebook E\n({compoonents_explained_variance_ratio*100 :.1f}% of the data variablility captured by three PCs)")
                    plt.savefig(self.main_folder_path + f"PCA_3D/{ str(epoch_id).zfill(digit_size) }_E_PCA_1st_2nd_3rd_PCs.png")
                plt.close()
        
        E = self.model.VQ.E.weight.data.detach().cpu() # dim = K x D matrix
        if E.shape[0] > 4:
            # get first top-10 components
            n_components = 5
            # PCA
            
            pca = PCA(n_components=n_components)
            E_pca = pca.fit(E).transform(E) # dim = K x 2
            cumsum_explained_variance_ratio_array = np.cumsum(pca.explained_variance_ratio_) 
            plt.figure()
            plt.plot(np.arange(1, 1 + n_components), cumsum_explained_variance_ratio_array)
            plt.xlabel('Number of PCs')
            plt.ylabel('Cumulative Explained Variance')
            plt.title(f'Cumulative Explained Variance (Epoch num. = {epoch_id})')
            if epoch_id == None:
                plt.legend([f'curve after {self.NUM_EPOCHS} epochs'])
            else:
                plt.legend([f'curve after {epoch_id} epochs'])
            
            plt.grid(True)
            
            if epoch_id == None:
                #plt.legend([f"epoch {e}" for e in np.arange( int(self.NUM_EPOCHS*0.05) ,self.NUM_EPOCHS + 1, int(self.NUM_EPOCHS*0.05))])
                plt.savefig(self.main_folder_path + f"9999_Cumulative_Explained_Variance_PCA.png")
                
                if self.PCA_decomp_in_every_epochs:
                    for format_ in format_list:
                        with imageio.get_writer(self.main_folder_path + f"Cumulative_Explained_Variance_PCA_{format_}.{format_}", mode='I',fps = frames_per_second) as writer:
                            for epoch_id_PCA in self.epoch_ids_PCA:
                                filename = self.main_folder_path + f"Cummulative_Explained_Variance_PCA/{ str(epoch_id_PCA).zfill(digit_size)}_Cumulative_Explained_Variance_PCA.png"
                                image = imageio.imread(filename)
                                writer.append_data(image)
                            writer.close()
            else:
                plt.savefig(self.main_folder_path + f"Cummulative_Explained_Variance_PCA/{ str(epoch_id).zfill(digit_size) }_Cumulative_Explained_Variance_PCA.png")
            plt.close()
            


    def codebook_visualization(self) -> None:
        if not self.model.args_VQ['train_with_quantization']:
            return None
        
        #self.plot_codebook_PCA(epoch_id = None)
        
        #LDA - requires labeled data
        if self.model.VQ.K != 2:
            # UMAP args
            n_neighbors = 3
            min_dist = 0.1
            #n_components = 2 #2D space for projection
            metric = 'cosine'
            E = self.model.VQ.E.weight.data.detach().cpu() # dim = K x D matrix
            #UMAP https://umap-learn.readthedocs.io/en/latest/parameters.html
            E_umap = umap.UMAP(n_neighbors=n_neighbors,
                                min_dist=min_dist, #n_components = n_components,
                                metric=metric).fit_transform(E)
            plt.figure()
            #plt.scatter(E_umap[:,0], E_umap[:,1], alpha=0.3)
            for i in range(self.model.VQ.K):
                plt.scatter(E_umap[i,0],
                            E_umap[i,1],
                            alpha=0.3,
                            color = 'k',
                            lw=1)#marker = target_name[i]
            
            plt.title("UMAP of the codebook E")
            plt.xlabel('First UMAP component')
            plt.ylabel('Second UMAP component')
            plt.grid()
            plt.savefig(self.main_folder_path + f"E_UMAP_1st_2nd_PCs.png")
            plt.close()
        
        #TSNE
        
        # PLOT PCA TSNE UMAP

    def visualize_interpolations(self):
        # Visualizing interpolations
        pass
    
    def visualize_reconstructions(self):
        # Visualizing reconstructions
        pass
    
    def visualize_discrete_codes(self, compose_transforms, dataset_str = 'test', create_plot_for_every_image_in_dataset = False, jupyter_show_images = False):
        assert(dataset_str in ['train', 'val', 'test'], f"The dataset can only be train, val or test and current dataset is {dataset_str}")
        
        self.compose_transforms = compose_transforms
        
        MAX_NUMBER_OF_IMAGES_TO_COVER = None#200#None#200#None
        
        # Visualizing the discrete codes from input images input-tensor of size (B, C, H, W)
        digit_size = len(str(len(self.loaders[dataset_str].dataset)))
        self.model.eval()

        self.visualize_tokens_path = self.main_folder_path + 'test_image_PLUS_encoding_counter_per_token_position_PLUS_histogram_of_codewords_usages/'
        if not os.path.exists(self.visualize_tokens_path):
            os.mkdir(self.visualize_tokens_path)
        
        # best images
        self.best_images_full_path = self.main_folder_path + 'test_image_PLUS_encoding_counter_per_token_position_PLUS_histogram_of_codewords_usages/best_imgs/'
        if not os.path.exists(self.best_images_full_path):
            os.mkdir(self.best_images_full_path)
        #clear the content of the folder
        files = glob.glob(self.best_images_full_path + '*')
        for f in files:
            os.remove(f)
                
        # worst images
        self.worst_images_full_path = self.main_folder_path + 'test_image_PLUS_encoding_counter_per_token_position_PLUS_histogram_of_codewords_usages/worst_imgs/'
        if not os.path.exists(self.worst_images_full_path):
            os.mkdir(self.worst_images_full_path)
            #clear the content of the folder
        files = glob.glob(self.worst_images_full_path + '*')
        for f in files:
            os.remove(f)
        
        # all images
        self.all_images_full_path =self.main_folder_path + 'test_image_PLUS_encoding_counter_per_token_position_PLUS_histogram_of_codewords_usages/_imgs/' 
        if not os.path.exists(self.all_images_full_path):
            os.mkdir(self.all_images_full_path)
        #clear the content of the folder
        files = glob.glob(self.all_images_full_path + '*')
        for f in files:
            os.remove(f)

        # for every image in the input batch plot three subplots on next to each other:
        # original image
        # Zq indices (discretized Ze)
        
        ############################    
        # visualize discrete codes #
        ############################

        # put the model to the specified device
        self.model = self.model.to(self.device)

        # put loaded model in the evaulation mode
        self.model.eval()
        
        #
        count_of_encoding_indices_per_token_position = torch.zeros((self.model.VQ.K, self.model.VQ.M+1,self.model.VQ.M+1), device='cpu')
        
        # counter 0,1,2,3.... across dataset
        index_ = -1
        
        # iterate image by image in the dataset
        for image_batch, image_id_batch in self.loaders[dataset_str]:
            if MAX_NUMBER_OF_IMAGES_TO_COVER != None:
                if index_ > MAX_NUMBER_OF_IMAGES_TO_COVER:
                    # hard stop at MAX_NUMBER_OF_IMAGES_TO_COVER-th image if MAX_NUMBER_OF_IMAGES_TO_COVER is defined (if MAX_NUMBER_OF_IMAGES_TO_COVER is not None)
                    break
            
            
            # inc. image counter by 1
            index_ += 1
            
            # it has to be image by image (because we are plotting image by image plots) hence batch-size has to be 1
            if self.loaders[dataset_str].batch_size != 1:
                assert(self.loaders[dataset_str].batch_size == 1, f"Mini-batch size of the test set should be 1, because of visualization and plotting later on in the code.")
            
            with torch.no_grad():
                # move the image tensor to the device
                image_batch = image_batch.to(self.device)
                
                # (VQ + Encoder) forward pass 
                # output every tensor in the vector quantization process 
                self.model.VQ.output_whole_quantization_process = True
                e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words, encoding_indices, estimate_codebook_words_freq, estimate_codebook_words_prob, inputs, D  = self.model.VQ(self.model.encoder(image_batch))
                
                self.model.VQ.output_whole_quantization_process = False
                
                #self.model.VQ(self.model.encoder(image_batch))
                
                e_and_q_latent_loss, encoding_indices, estimate_codebook_words_freq, estimate_codebook_words_prob, inputs, D = e_and_q_latent_loss.cpu(), encoding_indices.cpu(), estimate_codebook_words_freq.cpu(), estimate_codebook_words_prob.cpu(), inputs.cpu(), D.cpu()
                # e_and_q_latent_loss, e_latent_loss, q_latent_loss are scalars
                # Zq is BCHW tensor
                # estimate_codebook_words vector of size (K,)
                # encoding_indices vector of size (BHW,)
                encoding_indices_tensor = encoding_indices.view(-1, self.model.VQ.M+1, self.model.VQ.M+1)
                B = encoding_indices_tensor.size(0)
                #self.visualize_discrete_codes_output['count_codebook_words_per_token_position_used_in_' + dataset_str]
                
                count_of_encoding_indices_per_image = torch.zeros((self.model.VQ.M+1, self.model.VQ.M+1), device = 'cpu')

                for tokens_row_position in range(self.model.VQ.M + 1):
                    for tokens_column_position in range(self.model.VQ.M + 1):
                        unique_encoding_indices_per_image_per_position, count_of_unique_encoding_indices_per_image_per_position = torch.unique(input = encoding_indices_tensor[:, tokens_row_position, tokens_column_position], sorted=True, return_inverse=False, return_counts=True, dim=0)
                        
                        # this tensor count_of_encoding_indices_per_token_position is used later to produce other plots so do not deleete it!
                        count_of_encoding_indices_per_token_position[unique_encoding_indices_per_image_per_position.view(-1), tokens_row_position, tokens_column_position] += count_of_unique_encoding_indices_per_image_per_position.view(-1).float()
                        
                        #count_of_encoding_indices_per_image[unique_encoding_indices_per_image_per_position.view(-1), tokens_row_position, tokens_column_position] +=count_of_unique_encoding_indices_per_image_per_position.view(-1).float()
                        # TO DO OBAVEZNO !
                        count_of_encoding_indices_per_image[tokens_row_position, tokens_column_position] = unique_encoding_indices_per_image_per_position.view(-1)[0].float()

                if (create_plot_for_every_image_in_dataset) or (image_id_batch in self.best_imgs_ids) or (image_id_batch in self.worst_imgs_ids):
                    fig, axs = plt.subplots(1,4, figsize=(15,6), gridspec_kw={'width_ratios': [1, 1, 1, 1], 'height_ratios': [1]})
                    #matplotlib.rcParams.update(matplotlib.rcParamsDefault)
                    #plt.rcParams['text.usetex'] = True
                    #fig.set_size_inches(15, 7)
                    fig.suptitle(f"{index_+1}. image from {dataset_str} dataset", fontsize=20)

                    #plt.subplot(1, 3, 1)
                    # have to renormalize and show original image here
                    #image_batch_original_img = image_batch_original_img.view(1, image_batch_original_img.size(0), image_batch_original_img.size(1), image_batch_original_img.size(2))
                    image_batch_original_img = to_img(image_batch.cpu(), self.compose_transforms)

                    
                    
                    axs[0].imshow(np.transpose(image_batch_original_img[0,:,:,:], (1, 2, 0))) # H,W,C
                    
                    if self.config_path == "/home/novakovm/iris/MILOS/toy_shapes_config.yaml":
                        TOTAL_NUMBER_OF_SHAPES = get_hyperparam_from_config_file(self.config_path, "TOTAL_NUMBER_OF_SHAPES")
                        bin_id = (str(bin(image_id_batch))[2:]).zfill(14 * TOTAL_NUMBER_OF_SHAPES)
                        if TOTAL_NUMBER_OF_SHAPES == 2:
                            bin_id = bin_id[:len(bin_id)//2] + "\n" + bin_id[len(bin_id)//2:]
                        axs[0].set_title(f"Org. {image_batch_original_img.size(1)}x{image_batch_original_img.size(2)}x{image_batch_original_img.size(3)} image "\
                            + r"$X$" + f"dec= {str(image_id_batch.item()).zfill(digit_size)}\nbin= {bin_id}")
                    else:
                        axs[0].set_title(f"Org. {image_batch_original_img.size(1)}x{image_batch_original_img.size(2)}x{image_batch_original_img.size(3)} image "\
                            + r"$X$" + f"\nid= {str(image_id_batch.item()).zfill(digit_size)}")
                    axs[0].set_xlabel(r"$W$")
                    axs[0].set_ylabel(r"$H$")
                    #plt.subplot(1, 3, 2)
                    #different heatmaps
                    #axs[1].imshow(count_of_encoding_indices_per_image)
                    #axs[1].matshow(count_of_encoding_indices_per_image)
                    #axs[1].imshow(count_of_encoding_indices_per_image, cmap='hot', interpolation='nearest')
                    #axs[1] = 
                    sns.heatmap(data = count_of_encoding_indices_per_image.int().numpy(), linewidth=0.5, annot=True, ax=axs[1], cbar = False, xticklabels=np.arange(1,2 + self.model.VQ.M), yticklabels=np.arange(1,2 + self.model.VQ.M), square = True,  fmt='d', cmap='Blues')
                    #plt.show()
                    # different ways to show heatmap https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap

                    axs[1].set_title(f"The indices of codebook " + r"$E$" + f"\n({self.model.VQ.M+1}x{self.model.VQ.M+1} {self.model.VQ.D}-dim. vectors that form " + r"$Z_q$" + f" tensor)")
                    axs[1].set_xlabel(r"$W_e$")
                    axs[1].set_ylabel(r"$H_e$")
                    #axs[1].set_xticks(np.arange(0,1 + self.model.VQ.M))
                    #axs[1].set_yticks(np.arange(0,1 + self.model.VQ.M))
                    unique_encoding_indices_per_image, count_of_unique_encoding_indices_per_image = torch.unique(input = count_of_encoding_indices_per_image.view(-1), sorted=True, return_inverse=False, return_counts=True, dim=0)
                    histogram_unique_encoding_indices_per_image = np.zeros(self.model.VQ.K)
                    histogram_unique_encoding_indices_per_image[unique_encoding_indices_per_image.view(-1).numpy().astype(int)] = count_of_unique_encoding_indices_per_image.view(-1).numpy().astype(int)
                    histogram_unique_encoding_indices_per_image = 100*histogram_unique_encoding_indices_per_image/histogram_unique_encoding_indices_per_image.sum()
                    histogram_unique_encoding_indices_per_image = [float('nan') if x==0 else x for x in histogram_unique_encoding_indices_per_image]

                    #plot reconstruction
                    image_batch_recon  = self.model.decoder(Zq)
                    recon_error = (F.mse_loss(image_batch_recon, image_batch) / self.train_data_variance)
                    image_batch_recon_img = to_img(image_batch_recon.cpu(), self.compose_transforms)
                    axs[2].imshow(np.transpose(image_batch_recon_img[0,:,:,:], (1, 2, 0))) # H,W,C
                    axs[2].set_title(f"Reconstructed image " + r"$X_{rec}$" + " with loss\n"+ \
                                    r"$\mathcal{L}$" + f" = {(recon_error.item() + e_and_q_latent_loss.item() )*1e6 :.0f}" + r"$\cdot10^{-6}$=" + "\n" +\
                                    #f"\n(1/var)||X - X_rec ||^2 ({recon_error.item()*1e6 :.0f}e-6)"+\
                                    r"$\frac{1}{VAR[X_{train}]} ||X - X_{rec}||^2_2 $" + f" ( = {recon_error.item()*1e6 :.0f}" + r"$\cdot10^{-6}$" + f";{recon_error.item()/(recon_error.item() + e_and_q_latent_loss.item()) * 100. :.0f}%)\n" + \
                                    #f"\n+ (1+beta)*||Z_e-Z_q||^2 ({e_and_q_latent_loss.item()*1e6 :.0f}e-6)")
                                    r"$+(1+\beta) || \frac{Z_e}{||Z_e||_2} - \frac{Z_q}{||Z_q||_2}||^2_2 $" + f" ( = {e_and_q_latent_loss.item()*1e6 :.0f}" + r"$\cdot10^{-6}$" + f";{e_and_q_latent_loss.item()/(recon_error.item() + e_and_q_latent_loss.item()) * 100. :.0f}%)")
                    axs[2].set_xlabel(r"$W$")
                    axs[2].set_ylabel(r"$H$")
                    
                    # histogram of current tokens for a current image
                    markerline, stemline, baseline = axs[3].stem(np.arange(0, self.model.VQ.K), histogram_unique_encoding_indices_per_image, label = 'Token count')
                    plt.setp(stemline, linewidth = 1)
                    plt.setp(stemline, color = 'k')
                    plt.setp(markerline, markersize = 5)
                    plt.setp(markerline, color = 'k')
                    plt.setp(baseline, color = 'k')
                    
                    axs[3].plot(np.arange(0, self.model.VQ.K), np.ones(self.model.VQ.K) * (1. / (self.model.VQ.M + 1) ** 2) * 100, color = 'm', label = 'Uniform dist.')
                    axs[3].set_title(f"Percentage count of {(self.model.VQ.M+1)**2} tokens")
                    
                    #axs[3].legend(loc="upper left")

                    used_codewords = unique_encoding_indices_per_image.cpu().numpy().tolist()

                    #xticklabels = [ str(idx_) if idx_ in used_codewords else "" for idx_ in range(self.model.VQ.K)]
                    #xticklabels = [ ("\n"+x) if x_idx % 2 == 1 else x for x_idx, x in enumerate(xticklabels)]
                    
                    xticklabels = []
                    counter_ = 0
                    for idx_ in range(self.model.VQ.K):
                        if idx_ in used_codewords:
                            counter_ += 1
                            xticklabels.append(str(idx_) if counter_%2==1 else ("\n" + str(idx_)))
                        else:
                            xticklabels.append("")
                    axs[3].xaxis.set_ticks(np.arange(0, self.model.VQ.K))
                    axs[3].set_xticklabels(xticklabels, fontdict = {'fontsize' : 8})
                    plt.setp(axs[3].get_xticklabels(), rotation=30, horizontalalignment='right')

                    
                    # yticklabels = [y_value for y_value in histogram_unique_encoding_indices_per_image if not(np.isnan(y_value))] 
                    # yticklabels.append((1. / (self.model.VQ.M + 1) ** 2) * 100) #uniform dist value
                    # yticklabels = sorted(yticklabels)
                    # yticklabels = np.unique(np.array(yticklabels))
                    # yticklabels = [str(round(y_value,1)) for y_value in yticklabels]
                    # yticklabels = ["0"] + yticklabels # append minimum value of the plot, i.e., 0%
                    # yticklabels = yticklabels + ["40"] # append minimum value of the plot, i.e., 40%
                    # axs[3].set_yticklabels(yticklabels)
                    for x_,y_ in zip(np.arange(0, self.model.VQ.K), histogram_unique_encoding_indices_per_image):
                        if not(np.isnan(y_)):
                            axs[3].annotate( f"{y_: .0f}", xy=(x_,y_), xytext=(0,5), textcoords='offset points',ha='center')
                    
                    
                    axs[3].set_ylim(bottom=0, top=40)#the top is 40% this can be chaned ! (but if some token is used 40% or more then we are in a serious problem, I think)
                    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    
                    axs[3].grid(True)
                    
                    axs[3].set_xlabel(r"$K$" + f"-sized codebook indices")
                    axs[3].set_ylabel(f"Rounded percentage token count")
                    plt.tight_layout()

                    # save for all images
                    if create_plot_for_every_image_in_dataset:
                        fig.savefig(self.all_images_full_path + f"{str(index_+1).zfill(digit_size)}_th_test_image_PLUS_encoding_counter_per_token_position_PLUS_histogram_of_codewords_usages.png")
                        if jupyter_show_images:
                            fig.show()
                            return
                        
                    # do worst/best images split
                    if image_id_batch in self.best_imgs_ids:
                        index_in_best_array = self.best_imgs_ids.index(image_id_batch.item())
                        loss_of_index_in_best_array = self.best_imgs_losses[index_in_best_array]
                        fig.suptitle(f"{index_in_best_array+1}/{len(self.best_imgs_ids)} best reconstructed images (according to loss func. value) from {dataset_str} dataset", fontsize=20)
                        fig.savefig(self.best_images_full_path + f"{str(index_in_best_array+1).zfill(digit_size)}_th_best_{dataset_str}_img_loss_{int(round(loss_of_index_in_best_array*1e6,0))} e-6.png")

                    if image_id_batch in self.worst_imgs_ids:
                        index_in_worst_array = self.worst_imgs_ids.index(image_id_batch.item())
                        loss_of_index_in_worst_array = self.worst_imgs_losses[index_in_worst_array]
                        fig.suptitle(f"{index_in_worst_array+1}/{len(self.worst_imgs_ids)} worst reconstructed images (according to loss func. value) from {dataset_str} dataset", fontsize=20)
                        fig.savefig(self.worst_images_full_path + f"{str(index_in_worst_array+1).zfill(digit_size)}_th_worst_{dataset_str}_img_loss_{int(round(loss_of_index_in_worst_array*1e6,0))} e-6.png")
                    
                    plt.close()    
            
        # total number of codewords to code those images per codeword (1...K) in the codebook
        count_of_encoding_indices = count_of_encoding_indices_per_token_position.sum(dim=2).sum(dim=1).view(-1) # size=(K,)
        # total number of codewords to code those images is hence count_of_encoding_indices.sum()
        MAX_OCCURANCES_OF_SINGLE_ENCODING = count_of_encoding_indices.max().item()
        TOTAL_TOKEN_NUMBER = count_of_encoding_indices.sum().item() # = (number of images) * (M+1)^2
        
        
        #estimate probability of codewords usage
        prob_count_of_encoding_indices = count_of_encoding_indices/count_of_encoding_indices.sum()
        log_prob_count_of_encoding_indices = torch.log2(1e-12 + prob_count_of_encoding_indices)
        entropy_prob_count_of_encoding_indices = - torch.sum(prob_count_of_encoding_indices * log_prob_count_of_encoding_indices)
        perplexity_prob_count_of_encoding_indices = 2**(entropy_prob_count_of_encoding_indices)
        
        #entropy_report_str = r"$H(E)$"+ f" = {np.log2(self.model.VQ.K)} bits;  "+r"$\hat{H}(E)$"+f" = {entropy_prob_count_of_encoding_indices :.1f} bits ({entropy_prob_count_of_encoding_indices/np.log2(self.model.VQ.K)*100 :.1f} % of ideal)"
        #perplexity_report_str = r"$2^{H(E)}$"+f" = {self.model.VQ.K};  "+r"$2^{\hat{H}(E)}$"+f" = {perplexity_prob_count_of_encoding_indices :.1f} ({perplexity_prob_count_of_encoding_indices/self.model.VQ.K*100 :.1f} % of ideal)"

        entropy_report_str = r"$\hat{H}(E)$"+"/"+r"$H(E)$"+ f" = {entropy_prob_count_of_encoding_indices :.1f} / {np.log2(self.model.VQ.K)} bits ({entropy_prob_count_of_encoding_indices/np.log2(self.model.VQ.K)*100 :.1f}%)"
        perplexity_report_str = r"$2^{\hat{H}(E)}$"+"/"+r"$2^{H(E)}$"+f" = {perplexity_prob_count_of_encoding_indices :.1f} / {self.model.VQ.K} ({perplexity_prob_count_of_encoding_indices/self.model.VQ.K*100 :.1f}%)"

        
        
        fig, axs = plt.subplots(self.model.VQ.M + 1, self.model.VQ.M + 1, figsize=(13,13), gridspec_kw={'width_ratios': [1] * (self.model.VQ.M + 1) , 'height_ratios': [1] * (self.model.VQ.M + 1)}, sharex = True, sharey=True)
        title_ = f"Percentage of codebook indices used in the {int(TOTAL_TOKEN_NUMBER / (self.model.VQ.M + 1)**2)}-sized {dataset_str} dataset per token position\n" + entropy_report_str + "  " + perplexity_report_str + f"[unif. dist. {(1. / self.model.VQ.K) * 100 : .1f}%] "
        fig.suptitle(title_, fontsize=20)

        #count_of_encoding_indices_per_token_position
        # will use this as y-axis top limit
        MAX_PROBABILITY_PERCENTAGE = [count_of_encoding_indices_per_token_position[:, tokens_row_position, tokens_column_position].view(-1).cpu().numpy()  for tokens_row_position in range(self.model.VQ.M + 1) for tokens_column_position in range(self.model.VQ.M + 1)]
        MAX_PROBABILITY_PERCENTAGE = max([(x/x.sum()*100.).max() for x in MAX_PROBABILITY_PERCENTAGE])
        for tokens_row_position in range(self.model.VQ.M + 1): 
            for tokens_column_position in range(self.model.VQ.M + 1):
                x_axis_histogram_data = np.arange(self.model.VQ.K)
                y_axis_histogram_data = count_of_encoding_indices_per_token_position[:, tokens_row_position, tokens_column_position].view(-1).cpu().numpy()
                
                perplexity_ = 2**(-np.sum((y_axis_histogram_data/y_axis_histogram_data.sum()) * np.log2((1e-12 + y_axis_histogram_data/y_axis_histogram_data.sum()))))
                
                total_number_of_images = int(np.sum(y_axis_histogram_data))
                
                markerline, stemline, baseline = axs[tokens_row_position, tokens_column_position].stem(x_axis_histogram_data, y_axis_histogram_data / y_axis_histogram_data.sum() * 100, basefmt=" ", linefmt = ':', markerfmt = 'o')
                #plt.setp(stemline, linewidth = 1)
                #plt.setp(markerline, markersize = 2)
                #plt.setp(baseline, linewidth = 0)
                plt.setp(stemline, color = 'k')
                plt.setp(markerline, markersize = 5)
                plt.setp(markerline, color = 'k')
                plt.setp(baseline, color = 'k')
                
                
                
                axs[tokens_row_position, tokens_column_position].set_title(f"T[{tokens_row_position},{tokens_column_position}] "+  r"$2^{\hat{H}(E)}$" +f"={perplexity_ :.1f}")
                axs[tokens_row_position, tokens_column_position].set_xlim(left=0, right=self.model.VQ.K)
                axs[tokens_row_position, tokens_column_position].set_ylim(bottom=0, top=MAX_PROBABILITY_PERCENTAGE+2)#count_of_encoding_indices_per_token_position.max())
                
                axs[tokens_row_position, tokens_column_position].plot(np.arange(0, self.model.VQ.K), np.ones(self.model.VQ.K) * (1. / self.model.VQ.K) * 100, color = 'm') # np.ones(self.model.VQ.K) * (total_number_of_images * 1. / self.model.VQ.K)
                
                axs[tokens_row_position, tokens_column_position].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                axs[tokens_row_position, tokens_column_position].grid(True)
                
                max_number_of_annotations = 2
                max_value, is_max_value_annotated = np.sort((y_axis_histogram_data / y_axis_histogram_data.sum() * 100))[-1], False
                second_max_value, is_second_max_value_annotated = np.sort((y_axis_histogram_data / y_axis_histogram_data.sum() * 100))[-2], False
                for x_,y_ in zip(np.arange(0, self.model.VQ.K), y_axis_histogram_data / y_axis_histogram_data.sum() * 100):
                    #if y_ == (y_axis_histogram_data / y_axis_histogram_data.sum() * 100).max(): # only higest % elment
                    if y_ == max_value and not(is_max_value_annotated):
                        axs[tokens_row_position, tokens_column_position].annotate(f"({x_}, {y_:.0f}%)", xy=(x_,y_), xytext=(0,5), textcoords='offset points',ha='center')
                        
                        max_number_of_annotations -= 1
                        is_max_value_annotated = True
                        
                        if max_value == second_max_value:
                            is_second_max_value_annotated = True
                            max_number_of_annotations -= 1
                    
                    if y_ == second_max_value and not(is_second_max_value_annotated):
                        axs[tokens_row_position, tokens_column_position].annotate(f"({x_}, {y_:.0f}%)", xy=(x_,y_), xytext=(0,5), textcoords='offset points',ha='center')
                        max_number_of_annotations -= 1
                    
                        
                    if max_number_of_annotations == 0:
                        break
                        #break

                
                #axs[tokens_row_position, tokens_column_position].set_xlabel("Entries in the Codebook")
        # for ax in axs.flat:
        #     ax.set(xlabel = f"sorted {self.model.VQ.K}-codebook indices", 
        #         ylabel = f"count of codebook indices used in the {dataset_str} dataset")

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        
        fig.supxlabel(f"{self.model.VQ.K}-Entries (codewords) in the Codebook", fontsize=20)
        fig.supylabel("Percent of the used codeword in a specific token T position", fontsize=20)
        fig.savefig(self.main_folder_path + f"(M+1)_x_(M+1)_token_position_usage_across_{dataset_str}_dataset.png")
        plt.close()

        # occurances of codeword indices accross all positions
        # use this tensor -> count_of_encoding_indices

        fig,ax = plt.subplots(1, figsize=(8,8))


        markerline, stemline, baseline = ax.stem(np.arange(0, self.model.VQ.K), prob_count_of_encoding_indices * 100, label = 'Real Hist.', linefmt = ':', markerfmt = 'o')
        plt.setp(stemline, linewidth = 1)
        plt.setp(stemline, color = 'k')
        plt.setp(markerline, markersize = 5)
        plt.setp(markerline, color = 'k')
        plt.setp(baseline, color = 'k')

        ax.plot(np.arange(0, self.model.VQ.K), np.ones(self.model.VQ.K) * (1. / self.model.VQ.K) * 100, label = 'Ideal Hist. (Uniform Dist.)', color = 'm')

        #plt.setp(baseline, linewidth = 0)

        ax.legend()

        ax.set_title(f"Histogram in % of used tokens to quantize the WHOLE {dataset_str} dataset of {int(TOTAL_TOKEN_NUMBER / (self.model.VQ.M + 1)**2)} images\n" + entropy_report_str + " " + perplexity_report_str)

        used_codewords = unique_encoding_indices_per_image.cpu().numpy().tolist()

        xticklabels = [ str(idx_) if idx_ % 2 == 0 else ("\n"+str(idx_)) for idx_ in range(self.model.VQ.K)]

        #for idx_, xticklabel in enumerate(xticklabels):
        #    if idx_ in xticklabels:
        #        xticklabels[idx_] = str(idx_)
        ax.xaxis.set_ticks(np.arange(0, self.model.VQ.K))

        ax.set_xticklabels(xticklabels)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax.set_xlabel(f"Codewords entries in the Codebook")
        ax.set_ylabel(f"Percent of the codeword occurrence across all token position")
        
        for x_,y_ in zip(np.arange(0, self.model.VQ.K), prob_count_of_encoding_indices*100):
            # only annotate above uniform distrubition samples
            
            if y_>(1. / self.model.VQ.K) * 100:
                ax.annotate(f"({x_}, {y_:.0f}%)", xy=(x_,y_), xytext=(0,5), textcoords='offset points',ha='center')
                    
        

        plt.tight_layout()
        ax.grid(True)
        fig.savefig(self.main_folder_path + f"codewords_usage_across_{dataset_str}_dataset.png")
        plt.close()
        # # TO DO
        # # all names have to be unique
        # all_shape_features           = ['FILL_NOFILL',
        #                                 'SHAPE_TYPE_SPACE',
        #                                 'X_CENTER_SPACE',
        #                                 'Y_CENTER_SPACE',
        #                                 'COLOR_LIST',
        #                                 'a_CENTER_SPACE',
        #                                 'b_CENTER_SPACE',
        #                                 'alpha_CENTER_SPACE'
        #                                 ]
        # shape_feature_name_str = all_shape_features[0]
        # #histograms per position
        # for tokens_row_position in range(self.model.VQ.M + 1):
        #     for tokens_column_position in range(self.model.VQ.M + 1):
        

    
            
    def original_reconstructed_changed_reconstucted_tokens_changed_tokens(self,
                                                                        image_batch,# one image, it is the tensor of shape = (1,C,H,W)
                                                                        image_id_batch,#id
                                                                        new_encoding_indices, # new encoding_indices tensor of size (1, self.model.VQ.M+1, self.model.VQ.M+1) [where every element is an integer that is in the range 0, ..., (K - 1)] to generate a new Zq -> Zq_new so that a new image can be generated
                                                                        index_ = 0,
                                                                        dataset_str = 'test',
                                                                        create_plot_for_every_image_in_dataset = False, 
                                                                        jupyter_show_images = False):
        digit_size = len(str(len(self.loaders[dataset_str].dataset)))
        self.model.eval()
        
        
        #count_of_encoding_indices_per_token_position = torch.zeros((self.model.VQ.K, self.model.VQ.M+1,self.model.VQ.M+1), device='cpu')
        
        with torch.no_grad():
            # move the image tensor to the device
            image_batch = image_batch.to(self.device)
            
            # (VQ + Encoder) forward pass 
            # output every tensor in the vector quantization process 
            self.model.VQ.output_whole_quantization_process = True
            e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words, encoding_indices, estimate_codebook_words_freq, estimate_codebook_words_prob, inputs, D  = self.model.VQ(self.model.encoder(image_batch))
            
            self.model.VQ.output_whole_quantization_process = False
            
            #self.model.VQ(self.model.encoder(image_batch))
            
            e_and_q_latent_loss, encoding_indices, estimate_codebook_words_freq, estimate_codebook_words_prob, inputs, D = e_and_q_latent_loss.cpu(), encoding_indices.cpu(), estimate_codebook_words_freq.cpu(), estimate_codebook_words_prob.cpu(), inputs.cpu(), D.cpu()
            # e_and_q_latent_loss, e_latent_loss, q_latent_loss are scalars
            # Zq is BCHW tensor
            # estimate_codebook_words vector of size (K,)
            # encoding_indices vector of size (BHW,)
            #encoding_indices_tensor = encoding_indices.view(-1, self.model.VQ.M+1, self.model.VQ.M+1)
            #B = encoding_indices_tensor.size(0)
            #self.visualize_discrete_codes_output['count_codebook_words_per_token_position_used_in_' + dataset_str]
            
            """
            count_of_encoding_indices_per_image = torch.zeros((self.model.VQ.M+1, self.model.VQ.M+1), device = 'cpu')
            
            
            for tokens_row_position in range(self.model.VQ.M + 1):
                for tokens_column_position in range(self.model.VQ.M + 1):
                    unique_encoding_indices_per_image_per_position, count_of_unique_encoding_indices_per_image_per_position = torch.unique(input = encoding_indices_tensor[:, tokens_row_position, tokens_column_position], sorted=True, return_inverse=False, return_counts=True, dim=0)
                    
                    # this tensor count_of_encoding_indices_per_token_position is used later to produce other plots so do not deleete it!
                    # count_of_encoding_indices_per_token_position[unique_encoding_indices_per_image_per_position.view(-1), tokens_row_position, tokens_column_position] += count_of_unique_encoding_indices_per_image_per_position.view(-1).float()
                    
                    #count_of_encoding_indices_per_image[tokens_row_position, tokens_column_position] = unique_encoding_indices_per_image_per_position.view(-1)[0].float() # worse, I think
                    count_of_encoding_indices_per_image[tokens_row_position, tokens_column_position] = unique_encoding_indices_per_image_per_position.view(-1).float().item() # better, I think
                    
            """
        
            fig, axs = plt.subplots(1,5, figsize=(30,8), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1], 'height_ratios': [1]})
            #matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            #plt.rcParams['text.usetex'] = True
            #fig.set_size_inches(15, 7)
            fig.suptitle(f"{index_}. custom {dataset_str} image", fontsize=20)

            #plt.subplot(1, 3, 1)
            # have to renormalize and show original image here
            #image_batch_original_img = image_batch_original_img.view(1, image_batch_original_img.size(0), image_batch_original_img.size(1), image_batch_original_img.size(2))
            image_batch_original_img = to_img(image_batch.cpu(), self.compose_transforms)

            axs[0].imshow(np.transpose(image_batch_original_img[0,:,:,:], (1, 2, 0))) # H,W,C
            #axs[0].set_title(f"Original {image_batch_original_img.size(1)}x{image_batch_original_img.size(2)}x{image_batch_original_img.size(3)} image "\
            #    + r"$X$" + f"dec_id= {str(image_id_batch).zfill(5)}\nbin_id = {(str(bin(image_id_batch))[2:]).zfill(14)}")
            
            if self.config_path == "/home/novakovm/iris/MILOS/toy_shapes_config.yaml":
                    TOTAL_NUMBER_OF_SHAPES = get_hyperparam_from_config_file(self.config_path, "TOTAL_NUMBER_OF_SHAPES")
                    bin_id = (str(bin(image_id_batch))[2:]).zfill(14 * TOTAL_NUMBER_OF_SHAPES)
                    if TOTAL_NUMBER_OF_SHAPES == 2:
                        bin_id = bin_id[:len(bin_id)//2] + "\n" + bin_id[len(bin_id)//2:]
                    axs[0].set_title(f"Org. {image_batch_original_img.size(1)}x{image_batch_original_img.size(2)}x{image_batch_original_img.size(3)} image "\
                        + r"$X$" + f" dec= {str(image_id_batch.item()).zfill(digit_size)}\nbin= {bin_id}")
            else:
                axs[0].set_title(f"Org. {image_batch_original_img.size(1)}x{image_batch_original_img.size(2)}x{image_batch_original_img.size(3)} image "\
                    + r"$X$" + f"\nid= {str(image_id_batch.item()).zfill(digit_size)}")
                
            axs[0].set_xlabel(r"$W$")
            axs[0].set_ylabel(r"$H$")


            #plot reconstruction
            image_batch_recon  = self.model.decoder(Zq).cpu()
            recon_error = (F.mse_loss(image_batch_recon, image_batch.cpu()) / self.train_data_variance)
            image_batch_recon_img = to_img(image_batch_recon.cpu(), self.compose_transforms)
            axs[1].imshow(np.transpose(image_batch_recon_img[0,:,:,:], (1, 2, 0))) # H,W,C
            axs[1].set_title(f"Reconstructed image " + r"$X_{rec}$" + " with loss\n"+ \
                            r"$\mathcal{L}$" + f" = {(recon_error.item() + e_and_q_latent_loss.item() )*1e6 :.0f}" + r"$\cdot10^{-6}$=" + "\n" +\
                            #f"\n(1/var)||X - X_rec ||^2 ({recon_error.item()*1e6 :.0f}e-6)"+\
                            r"$\frac{1}{VAR[X_{train}]} ||X - X_{rec}||^2_2 $" + f" ( = {recon_error.item()*1e6 :.0f}" + r"$\cdot10^{-6}$" + f";{recon_error.item()/(recon_error.item() + e_and_q_latent_loss.item()) * 100. :.0f}%)\n" + \
                            #f"\n+ (1+beta)*||Z_e-Z_q||^2 ({e_and_q_latent_loss.item()*1e6 :.0f}e-6)")
                            r"$+(1+\beta) || \frac{Z_e}{||Z_e||_2} - \frac{Z_q}{||Z_q||_2}||^2_2 $" + f" ( = {e_and_q_latent_loss.item()*1e6 :.0f}" + r"$\cdot10^{-6}$" + f";{e_and_q_latent_loss.item()/(recon_error.item() + e_and_q_latent_loss.item()) * 100. :.0f}%)")
            axs[1].set_xlabel(r"$W$")
            axs[1].set_ylabel(r"$H$")
            
            
            
            #plot reconstruction of chaned Zq
            Zq_new = self.model.VQ.generate_new_Zq(new_encoding_indices = new_encoding_indices, B = 1)
            image_batch_recon_from_Zq_new  = self.model.decoder(Zq_new).cpu() 
            image_batch_recon_from_Zq_new_recon_error = (F.mse_loss(image_batch_recon_from_Zq_new, image_batch.cpu()) / self.train_data_variance)
            
            
            #recon_error = (F.mse_loss(image_batch_recon, image_batch) / self.train_data_variance)
            image_batch_recon_from_Zq_new_img = to_img(image_batch_recon_from_Zq_new.cpu(), self.compose_transforms)
            axs[2].imshow(np.transpose(image_batch_recon_from_Zq_new_img[0,:,:,:], (1, 2, 0))) # H,W,C
            axs[2].set_title("Reconstructed image "+ r"$\tilde{X}$" + " from the " + r"$\tilde{Z}_q$" + " token map\n" +\
                            r"$\frac{1}{VAR[X_{train}]} ||X - X_{rec}||^2_2 $" + f" = {image_batch_recon_from_Zq_new_recon_error.item()*1e6 :.0f}" + r"$\cdot10^{-6}$")
            # axs[2].set_title(f"Reconstructed image " + r"$X_{rec}$" + " with loss\n"+ \
            #                 r"$\mathcal{L}$" + f" = {(recon_error.item() + e_and_q_latent_loss.item() )*1e6 :.0f}" + r"$\cdot10^{-6}$=" + "\n" +\
            #                 #f"\n(1/var)||X - X_rec ||^2 ({recon_error.item()*1e6 :.0f}e-6)"+\
            #                 r"$\frac{1}{VAR[X_{train}]} ||X - X_{rec}||^2_2 $" + f" ( = {recon_error.item()*1e6 :.0f}" + r"$\cdot10^{-6}$" + f";{recon_error.item()/(recon_error.item() + e_and_q_latent_loss.item()) * 100. :.0f}%)\n" + \
            #                 #f"\n+ (1+beta)*||Z_e-Z_q||^2 ({e_and_q_latent_loss.item()*1e6 :.0f}e-6)")
            #                 r"$+(1+\beta) || \frac{Z_e}{||Z_e||_2} - \frac{Z_q}{||Z_q||_2}||^2_2 $" + f" ( = {e_and_q_latent_loss.item()*1e6 :.0f}" + r"$\cdot10^{-6}$" + f";{e_and_q_latent_loss.item()/(recon_error.item() + e_and_q_latent_loss.item()) * 100. :.0f}%)")
            axs[2].set_xlabel(r"$W$")
            axs[2].set_ylabel(r"$H$")
            
            
            
            
            
            #plt.subplot(1, 3, 2)
            #different heatmaps
            #axs[1].imshow(count_of_encoding_indices_per_image)
            #axs[1].matshow(count_of_encoding_indices_per_image)
            #axs[1].imshow(count_of_encoding_indices_per_image, cmap='hot', interpolation='nearest')
            #axs[1] = 
        
            #sns.heatmap(data = count_of_encoding_indices_per_image.int().numpy(), linewidth=0.5, annot=True, ax=axs[3], cbar = False, xticklabels=np.arange(1,2 + self.model.VQ.M), yticklabels=np.arange(1,2 + self.model.VQ.M), square = True,  fmt='d', cmap='Blues')
            sns.heatmap(data = encoding_indices.view(self.model.VQ.M+1, self.model.VQ.M+1).cpu().int().numpy(), linewidth=0.5, annot=True, ax=axs[3], cbar = False, xticklabels=np.arange(1,2 + self.model.VQ.M), yticklabels=np.arange(1,2 + self.model.VQ.M), square = True,  fmt='d', cmap='Blues')
            
            # different ways to show heatmap https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap

            axs[3].set_title(f"The original token map "+ r"$Z_q$"+" of the org. image " + r"$X$" + f"\n({self.model.VQ.M+1}x{self.model.VQ.M+1} {self.model.VQ.D}-dimensional vectors that form " + r"$Z_q$" + f" tensor)")
            axs[3].set_xlabel(r"$W_e$")
            axs[3].set_ylabel(r"$H_e$")
            #axs[1].set_xticks(np.arange(0,1 + self.model.VQ.M))
            #axs[1].set_yticks(np.arange(0,1 + self.model.VQ.M))
            
            #plt.subplot(1, 3, 2)
            #different heatmaps
            #axs[1].imshow(count_of_encoding_indices_per_image)
            #axs[1].matshow(count_of_encoding_indices_per_image)
            #axs[1].imshow(count_of_encoding_indices_per_image, cmap='hot', interpolation='nearest')
            #axs[1] = 
            sns.heatmap(data = new_encoding_indices.view(self.model.VQ.M+1, self.model.VQ.M+1).cpu().int().numpy(), linewidth=0.5, annot=True, ax=axs[4], cbar = False, xticklabels=np.arange(1,2 + self.model.VQ.M), yticklabels=np.arange(1,2 + self.model.VQ.M), square = True,  fmt='d', cmap='Blues')
            #plt.show()
            # different ways to show heatmap https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap

            axs[4].set_title(f"The changed token map " + r"$\tilde{Z}_q$")
            axs[4].set_xlabel(r"$W_e$")
            axs[4].set_ylabel(r"$H_e$")
            #axs[1].set_xticks(np.arange(0,1 + self.model.VQ.M))
            #axs[1].set_yticks(np.arange(0,1 + self.model.VQ.M))
            
            
            """"
            unique_encoding_indices_per_image, count_of_unique_encoding_indices_per_image = torch.unique(input = count_of_encoding_indices_per_image.view(-1), sorted=True, return_inverse=False, return_counts=True, dim=0)
            histogram_unique_encoding_indices_per_image = np.zeros(self.model.VQ.K)
            histogram_unique_encoding_indices_per_image[unique_encoding_indices_per_image.view(-1).numpy().astype(int)] = count_of_unique_encoding_indices_per_image.view(-1).numpy().astype(int)
            histogram_unique_encoding_indices_per_image = 100*histogram_unique_encoding_indices_per_image/histogram_unique_encoding_indices_per_image.sum()
            histogram_unique_encoding_indices_per_image = [float('nan') if x==0 else x for x in histogram_unique_encoding_indices_per_image]

            # histogram of current tokens for a current image
            markerline, stemline, baseline = axs[3].stem(np.arange(0, self.model.VQ.K), histogram_unique_encoding_indices_per_image, label = 'Token count')
            plt.setp(stemline, linewidth = 1)
            plt.setp(stemline, color = 'k')
            plt.setp(markerline, markersize = 5)
            plt.setp(markerline, color = 'k')
            plt.setp(baseline, color = 'k')
            
            axs[3].plot(np.arange(0, self.model.VQ.K), np.ones(self.model.VQ.K) * (1. / (self.model.VQ.M + 1) ** 2) * 100, color = 'm', label = 'Uniform dist.')
            axs[3].set_title(f"Percentage count of {(self.model.VQ.M+1)**2} tokens")
            
            #axs[3].legend(loc="upper left")
            

            used_codewords = unique_encoding_indices_per_image.cpu().numpy().tolist()

            #xticklabels = [ str(idx_) if idx_ in used_codewords else "" for idx_ in range(self.model.VQ.K)]
            #xticklabels = [ ("\n"+x) if x_idx % 2 == 1 else x for x_idx, x in enumerate(xticklabels)]
            
            xticklabels = []
            counter_ = 0
            for idx_ in range(self.model.VQ.K):
                if idx_ in used_codewords:
                    counter_ += 1
                    xticklabels.append(str(idx_) if counter_%2==1 else ("\n" + str(idx_)))
                else:
                    xticklabels.append("")
            axs[3].xaxis.set_ticks(np.arange(0, self.model.VQ.K))
            axs[3].set_xticklabels(xticklabels, fontdict = {'fontsize' : 8})
            plt.setp(axs[3].get_xticklabels(), rotation=30, horizontalalignment='right')

            
            # yticklabels = [y_value for y_value in histogram_unique_encoding_indices_per_image if not(np.isnan(y_value))] 
            # yticklabels.append((1. / (self.model.VQ.M + 1) ** 2) * 100) #uniform dist value
            # yticklabels = sorted(yticklabels)
            # yticklabels = np.unique(np.array(yticklabels))
            # yticklabels = [str(round(y_value,1)) for y_value in yticklabels]
            # yticklabels = ["0"] + yticklabels # append minimum value of the plot, i.e., 0%
            # yticklabels = yticklabels + ["40"] # append minimum value of the plot, i.e., 40%
            # axs[3].set_yticklabels(yticklabels)
            for x_,y_ in zip(np.arange(0, self.model.VQ.K), histogram_unique_encoding_indices_per_image):
                if not(np.isnan(y_)):
                    axs[3].annotate( f"{y_: .0f}", xy=(x_,y_), xytext=(0,5), textcoords='offset points',ha='center')
            
            
            axs[3].set_ylim(bottom=0, top=40)#the top is 40% this can be chaned ! (but if some token is used 40% or more then we are in a serious problem, I think)
            axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            
            axs[3].grid(True)
            
            axs[3].set_xlabel(r"$K$" + f"-sized codebook indices")
            axs[3].set_ylabel(f"Rounded percentage token count")
            plt.tight_layout()
            """
            fig.savefig(self.all_images_full_path + f"{str(index_).zfill(digit_size)}_custom_image.png")
            if jupyter_show_images:
                plt.show()
            plt.close()
            """
            # save for all images
            if create_plot_for_every_image_in_dataset:
                
                if jupyter_show_images:
                    fig.show()
                    return
                
            # do worst/best images split
            if image_id_batch in self.best_imgs_ids:
                index_in_best_array = self.best_imgs_ids.index(image_id_batch.item())
                loss_of_index_in_best_array = self.best_imgs_losses[index_in_best_array]
                fig.suptitle(f"{index_in_best_array+1}/{len(self.best_imgs_ids)} best reconstructed images (according to loss func. value) from {dataset_str} dataset", fontsize=20)
                fig.savefig(self.best_images_full_path + f"{str(index_in_best_array+1).zfill(digit_size)}_th_best_{dataset_str}_img_loss_{int(round(loss_of_index_in_best_array*1e6,0))} e-6.png")

            if image_id_batch in self.worst_imgs_ids:
                index_in_worst_array = self.worst_imgs_ids.index(image_id_batch.item())
                loss_of_index_in_worst_array = self.worst_imgs_losses[index_in_worst_array]
                fig.suptitle(f"{index_in_worst_array+1}/{len(self.worst_imgs_ids)} worst reconstructed images (according to loss func. value) from {dataset_str} dataset", fontsize=20)
                fig.savefig(self.worst_images_full_path + f"{str(index_in_worst_array+1).zfill(digit_size)}_th_worst_{dataset_str}_img_loss_{int(round(loss_of_index_in_worst_array*1e6,0))} e-6.png")
            
            plt.close()  
            """