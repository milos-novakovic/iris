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
        
        p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        
        # H : pad_top, pad_bottom,
        # W : pad_left, pad_right
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv1_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        
        x = F.pad(x, padding, "constant", 0)
        x = self.bn1(F.relu(self.conv1(x)))
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv2_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        
        x = F.pad(x, padding, "constant", 0)
        x = F.relu(self.conv2(x))
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv3_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, padding, "constant", 0)
        
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        #latent_tensor = x
        
        # decoder part
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv4_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, padding, "constant", 0)
        
        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        
        H_pad_top, H_pad_bottom, W_pad_left, W_pad_right = self.conv5_padding
        padding_left,padding_right, padding_top, padding_bottom = W_pad_left, W_pad_right,H_pad_top, H_pad_bottom
        padding = (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, padding, "constant", 0)
        
        x = torch.sigmoid(self.conv5(x))
        return x
        
        #latent = self.encoder(x)
        #x_recon = self.decoder(latent)
        #return x_recon