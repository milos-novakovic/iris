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


from models import *
# Hyper parameters
current_working_absoulte_path = '/home/novakovm/iris/MILOS'
os.chdir(current_working_absoulte_path)


H,W = 64, 64
args_train = {}
args_train['TOTAL_NUMBER_OF_IMAGES'] = 100000
args_test = {}
args_test['TOTAL_NUMBER_OF_IMAGES'] = 100000

num_workers = 4
#learning_rate = 0.001
#latent_dims = 10

capacity = 64

LATENT_DIM = 10

USE_GPU = True
TRAIN_FLAG = True

NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 3 * 1e-3#0.003

TRAIN_DATA_PATH = './DATA/'
TEST_DATA_PATH = './DATA_TEST/'

TRAIN_IMAGES_MEAN_FILE_PATH, TRAIN_IMAGES_STD_FILE_PATH = './DATA/RGB_mean.npy', './DATA/RGB_std.npy'



TRANSFORM_IMG = transforms.Compose([
#    transforms.Resize(256),
#    transforms.CenterCrop(256),
    #transforms.ToTensor(),
    transforms.Normalize(mean=np.load(TRAIN_IMAGES_MEAN_FILE_PATH).tolist(),
                         std=np.load(TRAIN_IMAGES_STD_FILE_PATH).tolist() )
    # OR
    # transforms.Normalize(mean = [0., 0., 0.],
    #                      std  = [255., 255., 255.])
    ])

# Train Data & Train data Loader
# Image Folder = A generic data loader where the images are arranged in this way by default


#train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data = CustomImageDataset(args = args_train, root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers)

# Test Data & Test data Loader
#test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data = CustomImageDataset(args = args_test, root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(dataset = test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers) 

# Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=TRAIN_DATA_PATH,
#                                            batch_size=batchsize,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=TEST_DATA_PATH,
#                                           batch_size=batchsize,
#                                           shuffle=False)


def conv2d_dims(h_in,w_in,k,s,p,d):
    h_out = np.floor( (h_in + 2 * p[0] - d[0] * (k[0] - 1) - 1 ) / s[0] + 1)
    w_out = np.floor( (w_in + 2 * p[1] - d[1] * (k[1] - 1) - 1 ) / s[1] + 1)
    return int(h_out), int(w_out)
            



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




autoencoder = Autoencoder(params_encoder=params_encoder, params_decoder=params_decoder)
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)
num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# set to training mode
autoencoder.train()

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
            image_batch_recon = autoencoder(image_batch)
            
            # reconstruction error
            loss = F.mse_loss(image_batch_recon, image_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, NUM_EPOCHS, train_loss_avg[-1]))#2h
        print(f'{epoch}th epoch took {round((time.time() - start_time_epoch),0)} seconds. ')
        
    
    current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time())) # 2022_11_19_20_11_26
    
    train_loss_avg_path = '/home/novakovm/iris/MILOS/train_loss_avg_' + current_time_str + '.npy'
    np.save(train_loss_avg_path,np.array(train_loss_avg))
    
    pretrained_autoencoder_path = '/home/novakovm/iris/MILOS/autoencoder_' + current_time_str + '.py'
    torch.save(autoencoder.state_dict(), pretrained_autoencoder_path)

# plt.ion()

# fig = plt.figure()
# plt.plot(train_loss_avg)
# plt.xlabel('Epochs')
# plt.ylabel('Reconstruction error')
# plt.show()

current_time_str = '2022_11_19_20_11_26'
autoencoder_loaded_path = '/home/novakovm/iris/MILOS/autoencoder_' + current_time_str + '.py'
autoencoder_loaded = Autoencoder()
autoencoder_loaded.load_state_dict(torch.load(autoencoder_loaded_path))
autoencoder_loaded.eval()


print('Testing ...')
test_loss_avg, num_batches = 0, 0
for image_batch, _ in test_data_loader:
    
    with torch.no_grad():

        image_batch = image_batch.to(device)

        # autoencoder reconstruction
        image_batch_recon = autoencoder(image_batch)

        # reconstruction error
        loss = F.mse_loss(image_batch_recon, image_batch)

        test_loss_avg += loss.item()
        num_batches += 1
    
test_loss_avg /= num_batches
print('average reconstruction error: %f' % (test_loss_avg))



#plt.ion()


autoencoder.eval()

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x):
    
    #np.save('./DATA/RGB_mean.npy', RGB_mean) 
    RGB_mean = np.load('./DATA/RGB_mean.npy')
    RGB_std = np.load('./DATA/RGB_std.npy')
    R_mean, G_mean, B_mean = RGB_mean[0], RGB_mean[1], RGB_mean[2]
    R_std, G_std, B_std = RGB_std[0], RGB_std[1], RGB_std[2]
    
    MIN_PIXEL_VALUE, MAX_PIXEL_VALUE = 0,255
    # red chanel of the image
    x[0, :, :] =  R_std * x[0, :, :] + R_mean
    x[0, :, :] = x[0, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    # green chanel of the image
    x[1, :, :] =  G_std * x[1, :, :] + G_mean
    x[1, :, :] = x[1, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    # blue chanel of the image
    x[2, :, :] =  B_std * x[2, :, :] + B_mean
    x[2, :, :] = x[2, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    
    x = np.round(x) #x = np.round(x*255.)
    x = x.astype(int)
    
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    return x

def show_image(img, path = './SHOW_IMAGES/show_image.png'):
    img = to_img(img) # C = size(0), H = size(1), W = size(2)
    img_np = img.numpy()
    plt.plot(np.transpose(img_np, (1, 2, 0))) # H,W,C
    plt.savefig(path)
    plt.close()
    #plt.imshow(np.transpose(npimg, (1, 2, 0))) # (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
    #cv2.imwrite(path, image) 

def visualise_output(images, model):

    with torch.no_grad():

        images = images.to(device)
        images = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        #plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.plot(np.transpose(np_imagegrid, (1, 2, 0))) # H,W,C
        path = './SHOW_IMAGES/visualise_output.png'
        #plt.show()
        plt.savefig(path)
        plt.close()
        

images, labels = iter(test_data_loader).next()

# First visualise the original images
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()

# Reconstruct and visualise the images using the autoencoder
print('Autoencoder reconstruction:')
visualise_output(images, autoencoder)



#'./data/MNIST_AE_pretrained/my_autoencoder.pth'

# this is how the autoencoder parameters can be saved:
#torch.save(autoencoder.state_dict(), pretrained_autoencoder_path) # OBAVEZNO!