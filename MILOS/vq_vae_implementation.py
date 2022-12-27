#from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import pandas as pd
from VQ_VAE import *

# EMPTY CACHE BEFORE RUNNING!
#torch.cuda.empty_cache()
def report_cuda_memory_status():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n" + f"Total Memory = {t/1e9 : .2f} GB.")
    print(f"Reserved Memory = {r/1e9 : .2f} GB.")
    print(f"Allocated inside Reserved Memory = {a/1e9 : .2f} GB.")
    print(f"Free inside Reserved Memory = {f/1e9 : .2f} GB.\n")

    global_free, total_gpu_memory_occupied = torch.cuda.mem_get_info('cuda:0') #for a given device 'cuda:' using cudaMemGetInfo  
    print(f'Global Free memory on the cuda:0 = {global_free/1e9 : .2f} GB.')
    print(f'Total GPU Memory occupied on the cuda:0 = {total_gpu_memory_occupied/1e9 : .2f} GB.\n')

# get the number of parameters
#from prettytable import PrettyTable

def count_parameters(model):
    #table = PrettyTable(["Modules", "Parameters"])
    Modules_list = []
    Parameters_list = []
    Parameters_list_percent = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        params = int(params/1e3)
        Modules_list.append(name)
        Parameters_list.append(params)
        #table.add_row([name, params])
        total_params+=params
    Parameters_list_percent = [round(param_ / total_params * 100.,2) for param_ in Parameters_list]
    table = pd.DataFrame({"Module Name" : Modules_list, "# of params in thousands" : Parameters_list, "# of params [%]" : Parameters_list_percent})
    print(table)
    print(f"Total Trainable Params in thousands: {total_params}")
    
    with open('log_all.txt', 'a') as f:
        f.write(f"\n Total Trainable Params in thousands: {total_params} \n")
        f.write(f"\n{table.to_string()}\n\n")
    
    return total_params


#################################
# Vector Quantization arguments #
#################################
C,H,W = 3,64,64
args_VQ = {}
K,D,run_id,M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),  int(sys.argv[4]) # K D run_id M

true_number_of_bits_per_image =14
compressed_number_of_bits_per_image = int(np.ceil(np.log2(K))) * (M + 1) ** 2
compression_gain = true_number_of_bits_per_image / compressed_number_of_bits_per_image
print(f"True # of bits per image        = {true_number_of_bits_per_image}")
print(f"Compressed # of bits per image  = {true_number_of_bits_per_image}")
print(f"So a reduction of {round(compression_gain,5)} bits is achieved")

# Folder renaming
# run_id = 0
# for d in np.array([6, 5, 4, 3, 2, 1]):
#     for k in np.array([64, 32, 16, 8]):
#         run_id += 1
#         compressed_number_of_bits_per_image = int(np.ceil(d * np.log2(k)))
#         trainer_folder_path_old = "/home/novakovm/iris/MILOS" + "/" + \
#                       str(run_id).zfill(3) + "_" + 'VQ_VAE' + \
#                       '_K_' + str(k) + \
#                       '_D_' + str(d)
#         trainer_folder_path_new = "/home/novakovm/iris/MILOS" + "/" + \
#                       str(run_id).zfill(3) + "_" + 'VQ_VAE' + \
#                       '_K_' + str(k) + \
#                       '_D_' + str(d) + \
#                       '_bits_' + str(compressed_number_of_bits_per_image)
#         old_name, new_name = trainer_folder_path_old, trainer_folder_path_new
#         os.system(f"mv {old_name} {new_name}")

args_VQ['D'] = D # 64 #embedding dimension
args_VQ['K'] = K # 512 #number of embeddings
args_VQ['beta'] = 0.25 #64 #embedding dimension
args_VQ['M'] = M
channel_number_in_hidden_layers = 128 #64 #128 #128 #256

#####################
# Encoder arguments # # floor[ (H + 2p - d(k-1) -1)/s + 1]
#####################

# with dilatation = 1 # floor[ (H + 2p -k )/s + 1]

# The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual
# 3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units. 

args_encoder = {}
args_encoder['M']=M #neccecary because of zero padding layer
args_encoder['C_in'], args_encoder['H_in'], args_encoder['W_in'] = C, H, W
args_encoder['conv1_Cin'] = args_encoder['C_in'] #Cin
args_encoder['conv1_Cout'] = channel_number_in_hidden_layers #64 #64
args_encoder['conv1_k'] = 4 #4
args_encoder['conv1_s'] = 2 #2
args_encoder['conv1_p'] = 1 #1
# floor[H/2]= H/2; if H is even (H_out = 32)

args_encoder['conv2_Cin'] = args_encoder['conv1_Cout']#64
args_encoder['conv2_Cout'] = channel_number_in_hidden_layers #128 #128
args_encoder['conv2_k'] = 4 #4
args_encoder['conv2_s'] = 2 #2
args_encoder['conv2_p'] = 1 #1
# floor[H/4] = H/4; if H/2 is even (H_out = 16)

args_encoder['conv3_Cin'] = args_encoder['conv2_Cout']#64
args_encoder['conv3_Cout'] = channel_number_in_hidden_layers #128 #128
args_encoder['conv3_k'] = 4 #4
args_encoder['conv3_s'] = 2 #2
args_encoder['conv3_p'] = 1 #1
# floor[H/8] = H/8; if H/4 is even (H_out = 8)

args_encoder['conv4_Cin'] = args_encoder['conv3_Cout']#64
args_encoder['conv4_Cout'] = channel_number_in_hidden_layers #128 #128
args_encoder['conv4_k'] = 4 #4
args_encoder['conv4_s'] = 2 #2
args_encoder['conv4_p'] = 1 #1
# floor[H/16] = H/16; if H/8 is even (H_out = 4)


# args_encoder['conv3_Cin'] = args_encoder['conv2_Cout'] #128
# args_encoder['conv3_Cout'] = 128 #128
# args_encoder['conv3_k'] = 3 #3
# args_encoder['conv3_s'] = 1 #1
# args_encoder['conv3_p'] = 1 #1
# floor[ H/4] = H/4; if H is div. by 4
# ([128, 128, 16, 16])     

# Same for Encoder and Decoder        
res_block_args={}
res_block_args['block_size'] = 2 #2
res_block_args['C_in'] = args_encoder['conv2_Cout'] #args_encoder['conv3_Cout'] # 128
res_block_args['C_mid'] = channel_number_in_hidden_layers#32

args_encoder['channel_adjusting_conv_Cin'] = res_block_args['C_in'] #C_enc, 128
args_encoder['channel_adjusting_conv_Cout'] = args_VQ['D'] #D,
args_encoder['channel_adjusting_conv_k'] = H//16 - args_VQ['M'] #H//4 #1 # kernel size has to be 1 because this (linear) conv layer just changes the chanell dimension
args_encoder['channel_adjusting_conv_s'] = 1
args_encoder['channel_adjusting_conv_p'] = 0
# Size = [B, D, M+1, M+1]

#####################
# Decoder arguments #
#####################

# The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions with stride
# 2 and window size 4 × 4. 


# >>> Ks = np.arange(1,17,1)
# >>> Ms = np.arange(0,16,1)
# >>> res = [(m,k, np.log2((62+k) / (m+k-1))) for m in Ms for k in Ks]
# >>> res = res[1:]
# >>> res1 = [x  for x in res if x[2] == int(x[2])]
# >>> res1
# m k n
# [(0, 2, 6.0), # Using this one! 
# (0, 10, 3.0), # Not using this one! <- but you can try it :)
# (1, 2, 5.0),
# (3, 2, 4.0),
# (6, 14, 2.0),
# (7, 2, 3.0), 
# (9, 10, 2.0),
# (12, 6, 2.0),
# (15, 2, 2.0)]
M_2_TRANS_CONV_KERNEL_SIZE, M_2_TRANS_CONV_LAYER_NUMBER = {}, {}
M_2_TRANS_CONV_KERNEL_SIZE[0], M_2_TRANS_CONV_LAYER_NUMBER[0] = 2, 6 # if M == 0
#M_2_TRANS_CONV_KERNEL_SIZE[0], M_2_TRANS_CONV_LAYER_NUMBER[0] = 10, 3 # if M == 0 # Not using this one! <- but you can try it :)
M_2_TRANS_CONV_KERNEL_SIZE[1], M_2_TRANS_CONV_LAYER_NUMBER[1] = 2, 5 # if M == 1
M_2_TRANS_CONV_KERNEL_SIZE[3], M_2_TRANS_CONV_LAYER_NUMBER[3] = 2, 4 # if M == 3
M_2_TRANS_CONV_KERNEL_SIZE[6], M_2_TRANS_CONV_LAYER_NUMBER[6] = 14, 2 # if M == 6

M_2_TRANS_CONV_KERNEL_SIZE[7], M_2_TRANS_CONV_LAYER_NUMBER[7] = 2, 3 # if M == 7
M_2_TRANS_CONV_KERNEL_SIZE[9], M_2_TRANS_CONV_LAYER_NUMBER[9] = 10, 2 # if M == 9
M_2_TRANS_CONV_KERNEL_SIZE[12], M_2_TRANS_CONV_LAYER_NUMBER[12] = 6, 2 # if M == 12
M_2_TRANS_CONV_KERNEL_SIZE[15], M_2_TRANS_CONV_LAYER_NUMBER[15] = 2, 2 # if M == 15

# if M is not M_2_TRANS_CONV_KERNEL_SIZE and M is not M_2_TRANS_CONV_LAYER_NUMBER:
#     assert(False, f"{M} is not in the M_2_TRANS_CONV_KERNEL_SIZE dict and M_2_TRANS_CONV_LAYER_NUMBER dict!")

TRANS_CONV_KERNEL_SIZE, TRANS_CONV_LAYER_NUMBER = M_2_TRANS_CONV_KERNEL_SIZE[M], M_2_TRANS_CONV_LAYER_NUMBER[M]

args_decoder = {}

args_decoder['channel_adjusting_conv_Cin'] = args_VQ['D'] #Cin
args_decoder['channel_adjusting_conv_Cout'] = channel_number_in_hidden_layers#128#128
args_decoder['channel_adjusting_conv_k'] = 1#H//4#1#3 #; TRY WITH KERNEL SIZE 1 AND PADDING 0
args_decoder['channel_adjusting_conv_s'] = 1
args_decoder['channel_adjusting_conv_p'] = 0#1 #; TRY WITH KERNEL SIZE 1 AND PADDING 0

# Same for Encoder and Decoder        
res_block_args={}
res_block_args['block_size'] = 2 #2
res_block_args['C_in'] = args_decoder['channel_adjusting_conv_Cout'] # 128
res_block_args['C_mid'] = channel_number_in_hidden_layers#32

#CONV_LAYERS = 4#int(np.log2(H))
args_decoder['CONV_LAYERS'] = 4#TRANS_CONV_LAYER_NUMBER
for i in range(1,args_decoder['CONV_LAYERS']+1):
    args_decoder[f'trans_conv{i}_Cin'] = args_decoder['channel_adjusting_conv_Cout'] #128
    args_decoder[f'trans_conv{i}_k'] = 4#TRANS_CONV_KERNEL_SIZE#4#8#4 #4
    args_decoder[f'trans_conv{i}_s'] = 2#8#2
    args_decoder[f'trans_conv{i}_p'] = 0#1#1
    
    if i == args_decoder['CONV_LAYERS']:
        args_decoder[f'trans_conv{i}_Cout'] = 3 # last conv layer C_out
        args_decoder[f'trans_conv{i}_k'] += 2 #8#4 #4
    else:
        args_decoder[f'trans_conv{i}_Cout'] = channel_number_in_hidden_layers#64#64

# args_decoder['trans_conv2_Cin'] = args_decoder['trans_conv1_Cout']#64
# args_decoder['trans_conv2_Cout'] = 3#3
# args_decoder['trans_conv2_k'] = 4#4
# args_decoder['trans_conv2_s'] = 2#2
# args_decoder['trans_conv2_p'] = 1#1
#torch.Size([1, 3, 4, 4])


#########################
# VQ-VAE model creation #
#########################

vq_vae_implemented_model = VQ_VAE(args_encoder, args_VQ, args_decoder, res_block_args)
    
count_parameters(vq_vae_implemented_model)

print(vq_vae_implemented_model(torch.empty(1,3,64,64).normal_())[1].size())

report_cuda_memory_status()

d=0
# We use the ADAM optimiser [21] with learning rate 2e-4 and evaluate
# the performance after 250,000 steps with batch-size 128. For VIMCO we use 50 samples in the
# multi-sample training objective.