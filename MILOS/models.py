import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms


import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, args, root = False, transform=False, target_transform=False):
        #self.img_labels = pd.read_csv(annotations_file)
        self.root = root # './DATA/' = '/home/novakovm/iris/MILOS/DATA/'
        self.transform = transform
        self.TOTAL_NUMBER_OF_IMAGES = args['TOTAL_NUMBER_OF_IMAGES']
        #self.target_transform = target_transform

    def __len__(self):
        return self.TOTAL_NUMBER_OF_IMAGES

    def __getitem__(self, idx):
        img_path = self.root + 'color_img_' + str(idx).zfill(len(str(self.TOTAL_NUMBER_OF_IMAGES))) + '.png'#os.path.join(self.root, self.img_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path).float() # .double() = torch.float64 and  .float() = torch.float32
        #label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image#, label

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
            self.conv2 = nn.Conv2d( in_channels=params_decoder['in_channels_conv2'],  # 2*64
                                    out_channels=params_decoder['out_channels_conv2'], # 64
                                    kernel_size=params_decoder['kernel_size_conv2'], # 4
                                    stride=params_decoder['stride_conv2'], # 2
                                    padding=params_decoder['padding_conv2'], # 1
                                    dilation= params_decoder['dilation_conv2']) # 1
            
            self.conv2_H_out = params_decoder['conv2_H_out']
            self.conv2_W_out = params_decoder['conv2_W_out']
        
        #self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        if params_decoder['conv1_exists']:
            self.conv1 = nn.Conv2d( in_channels=params_decoder['out_channels_conv2'],  # 64 = params_decoder['in_channels_conv1']
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