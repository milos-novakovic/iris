import yaml
from yaml.loader import SafeLoader
import itertools
from collections import OrderedDict
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import time

import os
import pandas as pd
from torchvision.io import read_image

from vq_vae_implementation import *

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, args, root = False, transform=False, target_transform=False):
        #self.img_labels = pd.read_csv(annotations_file)
        self.root = root # './DATA/' = '/home/novakovm/iris/MILOS/DATA/'
        self.transform = transform
        self.TOTAL_NUMBER_OF_IMAGES = args['TOTAL_NUMBER_OF_IMAGES'] # 2048 for test and val; 12'288 for train
        self.image_ids = args['image_ids']# 3314, 2151, 12030, 32, ...
        
        
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)#self.TOTAL_NUMBER_OF_IMAGES

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.root + 'color_img_' + str(image_id).zfill(len(str(self.TOTAL_NUMBER_OF_IMAGES))) + '.png'#os.path.join(self.root, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path).float() # .double() = torch.float64 and  .float() = torch.float32
        #label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
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
        super(My_VQ_VAE_Encoder, self).__init__()
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
        
        # create and init self.optimizer
        if self.optimizer_settings['optimization_algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.optimizer_settings['lr'])
            
        elif self.optimizer_settings['optimization_algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(params = self.model.parameters(), lr = self.optimizer_settings['lr'])
            
        else:
            assert(False, f"There are no optimizers called {self.optimizer_settings['optimization_algorithm']}.")

    def get_intermediate_training_stats_str(  self,\
                                            current_epoch, \
                                            total_nb_epochs, \
                                            train_duration_sec, \
                                            val_duration_sec, \
                                            start_time_training,\
                                            batch_size_train,\
                                            batch_size_val,\
                                            current_avg_train_loss,\
                                            current_avg_val_loss, \
                                            min_avg_train_loss, \
                                            min_avg_val_loss) -> str:
        
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
        
        
        intermediate_training_stats : str = \
                 f"Epoch {current_epoch+1}/{total_nb_epochs};\n"\
                +f"Training   Samples Mini-Batch size      = {batch_size_train};\n"\
                +f"Validation Samples Mini-Batch size      = {batch_size_val};\n"\
                +f"Total elapsed time in Training Epoch    = {train_duration_sec_str};\n"\
                +f"Total elapsed time in Validation Epoch  = {val_duration_sec_str};\n"\
                +f"Total elapsed time from begining        = {duration_sec_str};\n"\
                +f"Curr. Avg. Train Loss across Mini-Batch = {current_avg_train_loss *1e6 : .1f} e-6;\n"\
                +f"Curr. Avg. Val   Loss across Mini-Batch = {current_avg_val_loss *1e6 : .1f} e-6;\n"\
                +f"Min.  Avg. Train Loss across Mini-Batch = {min_avg_train_loss *1e6 : .1f} e-6;\n"\
                +f"Min.  Avg. Val   Loss across Mini-Batch = {min_avg_val_loss *1e6 : .1f} e-6;\n"\
                +f"\n----------------------------------------------------------------------------------\n"
        
        return intermediate_training_stats
    
    def train(self) -> None:
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
        
        # training and validation duration in seconds per epoch
        self.train_duration_per_epoch_seconds = [None]*self.NUM_EPOCHS
        self.val_duration_per_epoch_seconds = [None]*self.NUM_EPOCHS
        
        # epochs loop
        for epoch in range(self.NUM_EPOCHS):
            
            ###################
            # train the model #
            ###################
            
            # time when the training epoch started
            start_time_epoch = time.time()
            
            # init the avaraged training loss
            self.train_loss_avg.append(0.)
            
            # init the number of mini-batches covered in the current epoch
            num_batches = 0
            
            # init current training loss
            self.train_loss = 0.0
            
            # set the model to the train state
            self.model.train()
            
            for image_batch, image_ids_batch  in self.loaders['train']:
                
                #torch.Size([BATCH_SIZE_TRAIN, 3 (RGB), H, W])
                image_batch = image_batch.to(self.device)
                
                # autoencoder reconstruction
                # forward pass: compute predicted outputs by passing inputs to the model
                image_batch_recon = self.model(image_batch)
                
                # reconstruction error: calculate the batch loss
                if len(image_batch_recon) == 3:
                    vq_loss, image_batch_recon_, perplexity = image_batch_recon
                    recon_error = F.mse_loss(image_batch_recon_, image_batch)# / data_variance
                    loss = recon_error + vq_loss
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
                
                # count the number of batches
                num_batches += 1
                
            # calculate the current avg. training loss
            self.train_loss_avg[-1] /= (1.*num_batches)
            
            # calculate the current min. avg. training loss
            self.min_train_loss = np.min([self.min_train_loss, self.train_loss_avg[-1]])
            
            # calculate training elapsed time in the current epoch
            self.train_duration_per_epoch_seconds[epoch] = int(time.time() - start_time_epoch)
            
            ######################    
            # validate the model #
            ######################
            
            # time when the validation epoch started
            start_time_epoch = time.time()
            
            # init the avaraged validation loss
            self.val_loss_avg.append(0.)
            
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
                if len(image_batch_recon) == 3:
                    vq_loss, image_batch_recon_, perplexity = image_batch_recon
                    recon_error = F.mse_loss(image_batch_recon_, image_batch)# / data_variance
                    loss = recon_error + vq_loss
                else:
                    loss = self.loss_fn(image_batch_recon, image_batch)
                    
                
                # since this is validation of the model there is not (backprogagation and update step size)
                  
                # sum up the current validation loss in the last element of the train_loss_avg
                self.val_loss_avg[-1] += loss.item()
                
                # count the number of batches
                num_batches += 1
            
            
            # calculate the current avg. validation loss
            self.val_loss_avg[-1] /= (1.*num_batches)
            
            # calculate the current min. avg. validation loss
            self.min_val_loss = np.min([self.min_val_loss, self.val_loss_avg[-1]])
            
            # calculate validation elapsed time in the current epoch
            self.val_duration_per_epoch_seconds[epoch] = int(time.time() - start_time_epoch)
            
            # every 10th epoch print intermediate training/validation statistics
            if (epoch+1) % 10 == 0:
                print(self.get_intermediate_training_stats_str(\
                    current_epoch           = epoch, \
                    total_nb_epochs         = self.NUM_EPOCHS, \
                    train_duration_sec      = self.train_duration_per_epoch_seconds[epoch], \
                    val_duration_sec        = self.val_duration_per_epoch_seconds[epoch], \
                    start_time_training     = START_TIME_TRAINING, \
                    batch_size_train        = self.loaders['train'].batch_size,\
                    batch_size_val          = self.loaders['val'].batch_size,\
                    current_avg_train_loss  = self.train_loss_avg[-1],\
                    current_avg_val_loss    = self.val_loss_avg[-1], \
                    min_avg_train_loss      = self.min_train_loss, \
                    min_avg_val_loss        = self.min_val_loss\
                ))

        # get current time in the format YYYY_MM_DD_hh_mm_ss
        self.current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())) # 2022_11_19_20_11_26

        #train_loss_avg_path = '/home/novakovm/iris/MILOS/autoencoder_train_loss_avg_' + current_time_str + '.npy'
        
        # create training/validation avg. loss file paths
        self.train_loss_avg_path = self.main_folder_path + '/' + self.model_name + '_train_loss_avg_' + self.current_time_str + '.npy'
        self.val_loss_avg_path = self.main_folder_path + '/' + self.model_name + '_val_loss_avg_' + self.current_time_str + '.npy'

        # create training/validation avg. losses as numpy arrays to the above specified file paths
        self.train_loss_avg = np.array(self.train_loss_avg)
        self.val_loss_avg = np.array(self.val_loss_avg)
        np.save(self.train_loss_avg_path,
                self.train_loss_avg
                )
        np.save(self.val_loss_avg_path,
                self.val_loss_avg
                )
        
        # Message that the training/validation avg. losses are saved and the corresponding paths
        print(f"Autoencoder Training Loss Average saved here\n{self.train_loss_avg_path}", end = '\n\n')
        print(f"Autoencoder Validation Loss Average saved here\n{self.val_loss_avg_path}", end = '\n\n')

        # example of self.model_path:=            
        #/home/novakovm/iris/MILOS
        # /
        #vanilla_autoencoder
        # 2022_12_03_19_39_08
        #.py
        self.model_path = self.main_folder_path + '/' + self.model_name + self.current_time_str + '.py'
        
        # saving model trained parameters, so that it could be used as a pretrained model in the future usages
        torch.save(self.model.state_dict(),self.model_path)
        print(f"Current Trained Model saved at = \n {self.model_path}", end = '\n\n')
        
        TOTAL_TRAINING_TIME = int(time.time() - START_TIME_TRAINING)
        m, s = divmod(TOTAL_TRAINING_TIME, 60)
        h, m = divmod(m, 60)
        TOTAL_TRAINING_TIME = f"{h}:{m}:{s} h/m/s"
        
        print(f"Total training time is = {TOTAL_TRAINING_TIME}, end = '\n\n'")
        print("Training Ended", end = '\n--------------------------------------------------------------\n')
    
    def load_model(self, current_time_str, autoencoder_config_params_wrapped_sorted) -> None:
        #current time in the format YYYY_MM_DD_hh_mm_ss
        self.current_time_str  = current_time_str
        self.model_path = self.main_folder_path + '/' + self.model_name + self.current_time_str + '.py'
        
        # load model architecture in params wrapped and sorted fashion
        self.autoencoder_config_params_wrapped_sorted = autoencoder_config_params_wrapped_sorted
        
        # create a model (constructor)
        #self.model = Vanilla_Autoencoder_v02(autoencoder_config_params_wrapped_sorted=self.autoencoder_config_params_wrapped_sorted)
        self.model = vq_vae_implemented_model
        
        # load the model state from the model path
        self.model.load_state_dict(torch.load(self.model_path))
        
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
        self.test_loss = []

        # test_samples_loss is pandas DataFrame that has two columns
        # first column name is the test_image_id that is the id of the test image (int type)
        # second column name is the test_image_rec_loss that is the reconstruction loss of the test image with the id equal to test_image_id (float type)
        self.test_samples_loss = {} # test_image_tensor: test_loss
        self.test_samples_loss['test_image_id'] = []
        self.test_samples_loss['test_image_rec_loss'] = []

        # put the model to the specified device
        self.model = self.model.to(self.device)

        # put the loss to the specified device
        self.loss_fn.to(self.device)

        # put loaded model in the evaulation mode
        self.model.eval()

        for image_batch, image_id_batch in self.loaders['test']:
            assert(self.loaders['test'].batch_size == 1, f"Mini-batch size of the test set should be 1, because of visualization and plotting later on in the code.")
            
            # remember the test_image_id (i.e. id of the test image)
            self.test_samples_loss['test_image_id'].append(image_id_batch.item())
                
            with torch.no_grad():
                # move the image tensor to the device
                image_batch = image_batch.to(self.device)
                
                # autoencoder reconstruction (forward pass)
                image_batch_recon = self.model(image_batch)

                # reconstruction error (loss calculation)
                if len(image_batch_recon) == 3:
                    vq_loss, image_batch_recon_, perplexity = image_batch_recon
                    recon_error = F.mse_loss(image_batch_recon_, image_batch)# / data_variance
                    loss = recon_error + vq_loss
                else:
                    loss = self.loss_fn(image_batch_recon, image_batch)
                

                # remember the test_image_id's reconstruction loss (in a different array used for complex plotting)
                self.test_samples_loss['test_image_rec_loss'].append(loss.item())
                
                # remember the test_image_id's reconstruction loss (in a different array used for simple plotting)
                self.test_loss.append(loss.item())

        # cast it to np.array type 
        self.test_loss = np.array(self.test_loss)
        print(f'Average reconstruction error: {np.round(np.mean(self.test_loss)*1e6,1)} e-6')
        print("Testing Ended")

    def plot(self, train_val_plot = True, test_plot = True) -> None:
        # Plot Training and Validation Average Loss per Epoch
        if train_val_plot:
            plt.figure()
            # SEMILOG-Y SCALE for both validation and training loss
            plt.semilogy(self.train_loss_avg)
            plt.semilogy(self.val_loss_avg)
            plt.title(f'Train (Min. = {self.train_loss_avg.min() *1e3: .2f} e-3) & '\
                    + f'Validation (Min. = {self.val_loss_avg.min() *1e3: .2f} e-3) \n '\
                    + f'Loss Averaged across Mini-Batch per epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Mini-Batch Avg. Train & Validation Loss')
            plt.legend(['Training Loss','Validation Loss'])
            plt.grid()
            plt.savefig(self.main_folder_path + '/semilog_train_val_loss_per_epoch.png')
            plt.close()
            
        if test_plot:
            # Plot Test Loss for every sample in the Test set
            fig = plt.figure()
            ax = plt.gca()
            x_scatter = np.array(self.test_samples_loss['test_image_id'])
            y_scatter = np.array(self.test_samples_loss['test_image_rec_loss'])
            colors = "blue"#[5 for img_id in x_scatter]
            area = 20 #(30 * np.random.rand(N))**2  # 0 to 15 point radii
            plt.scatter(x_scatter, y_scatter, s=area, c=colors, alpha=0.4)
            ax.set_yscale('log')
            plt.title(f'Test Loss per sample in the Test set \n'+
                        f'(Avg. = {y_scatter.mean()*1e3 : .2f} e-3)')
            plt.grid()
            plt.xlabel('Test sample ids')
            plt.ylabel('Testing Loss')
            plt.savefig(self.main_folder_path + '/testing_loss_per_image_in_minibatch.png')
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
        
        x_scatter = np.array(self.test_samples_loss['test_image_id'])
        y_scatter = np.array(self.test_samples_loss['test_image_rec_loss'])
        
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
        plt.savefig(self.main_folder_path +\
        f"/SHOW_IMAGES/Test_Loss_Over_Test_Image_IDs_based_on_following_feature_shapes_{plot_title}.png")
        plt.close()
            
    def get_worst_test_samples(self, TOP_WORST_RECONSTRUCTED_TEST_IMAGES) -> None:
        # Visualization of top worst reconstructed test images (i.e. where autoencoder fails) 
        self.df_test_samples_loss = pd.DataFrame(self.test_samples_loss)
        self.df_test_samples_loss = self.df_test_samples_loss.sort_values('test_image_rec_loss',ascending=False)\
                                                            .reset_index(drop=True)
        #pick top- TOP_WORST_RECONSTRUCTED_TEST_IMAGES worst reconstructed images
        self.df_worst_reconstructed_test_images = self.df_test_samples_loss.head(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)
        print(f"pick top {TOP_WORST_RECONSTRUCTED_TEST_IMAGES} worst reconstructed images\n", self.df_worst_reconstructed_test_images.to_string())


        self.top_images, self.imgs_ids , self.imgs_losses = [], [], []
        for worst_reconstructed_test_image_id, worst_reconstructed_test_image_loss in zip(self.df_worst_reconstructed_test_images['test_image_id'], self.df_worst_reconstructed_test_images['test_image_rec_loss']):
            # find the test image index when you have test image id in the test_data.image_ids tha
            worst_reconstructed_test_image_id_index = np.where(self.loaders['test'].dataset.image_ids == worst_reconstructed_test_image_id)[0][0]
            
            # get the actual image as well as the image_id
            image, image_id = self.loaders['test'].dataset[worst_reconstructed_test_image_id_index]
            
            # save the test image (tensor)
            self.top_images.append(image)
            
            # save the test image id
            self.imgs_ids.append(image_id)
            
            # save the test image reconstruction error (i.e. loss value)
            self.imgs_losses.append(worst_reconstructed_test_image_loss)

        # saved top_images are list of tensor, so cast to a tensor with torch.stack() function
        self.top_images = torch.stack(self.top_images) #torch.Size(TOP_WORST_RECONSTRUCTED_TEST_IMAGES, C, H, W)
    
    def get_best_test_samples(self, TOP_BEST_RECONSTRUCTED_TEST_IMAGES) -> None:
        # Visualization of top best reconstructed test images (i.e. where autoencoder succeeds) 
        self.df_test_samples_loss = pd.DataFrame(self.test_samples_loss)
        self.df_test_samples_loss = self.df_test_samples_loss.sort_values('test_image_rec_loss',ascending=True)\
                                                            .reset_index(drop=True)
        #pick top- TOP_BEST_RECONSTRUCTED_TEST_IMAGES best reconstructed images
        self.df_best_reconstructed_test_images = self.df_test_samples_loss.head(TOP_BEST_RECONSTRUCTED_TEST_IMAGES)
        print(f"pick top {TOP_BEST_RECONSTRUCTED_TEST_IMAGES} best reconstructed images\n", self.df_best_reconstructed_test_images.to_string())


        self.top_images, self.imgs_ids , self.imgs_losses = [], [], []
        for best_reconstructed_test_image_id, best_reconstructed_test_image_loss in zip(self.df_best_reconstructed_test_images['test_image_id'], self.df_best_reconstructed_test_images['test_image_rec_loss']):
            # find the test image index when you have test image id in the test_data.image_ids tha
            best_reconstructed_test_image_id_index = np.where(self.loaders['test'].dataset.image_ids == best_reconstructed_test_image_id)[0][0]
            
            # get the actual image as well as the image_id
            image, image_id = self.loaders['test'].dataset[best_reconstructed_test_image_id_index]
            
            # save the test image (tensor)
            self.top_images.append(image)
            
            # save the test image id
            self.imgs_ids.append(image_id)
            
            # save the test image reconstruction error (i.e. loss value)
            self.imgs_losses.append(best_reconstructed_test_image_loss)

        # saved top_images are list of tensor, so cast to a tensor with torch.stack() function
        self.top_images = torch.stack(self.top_images) #torch.Size(TOP_WORST_RECONSTRUCTED_TEST_IMAGES, C, H, W)