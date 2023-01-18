import yaml
import numpy as np
import time
import os
import glob
from find_mean_std import find_mean_std
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Model_Trainer
import torch.nn.functional as F
from VQ_VAE import VQ_VAE, count_parameters, report_cuda_memory_status # ResidualStack, VectorQuantizer
from models import visualise_output, CustomImageDataset

import matplotlib.pyplot as plt
import torchvision
from helper_functions import get_hyperparam_from_config_file
# hyperparameters related to images
# get_hyperparam_from_config_file = lambda data_dict, hyperparam_name : \
#                             [dict_[hyperparam_name] for dict_ in data_dict
#                              if hyperparam_name in dict_][0]
def inference(config_path = "/home/novakovm/iris/MILOS/crafter_config.yaml"):                      
    # number of images for training and testing datasets

    H, W, C = get_hyperparam_from_config_file(config_path, 'H'), get_hyperparam_from_config_file(config_path, 'W'), get_hyperparam_from_config_file(config_path, 'C')

    NUM_EPOCHS =                            get_hyperparam_from_config_file(config_path, 'NUM_EPOCHS')
    NUM_WORKERS =                           get_hyperparam_from_config_file(config_path, 'NUM_WORKERS') # see what this represents exactly!
    USE_PRETRAINED_MODEL  =                 get_hyperparam_from_config_file(config_path, 'USE_PRETRAINED_MODEL')
    USE_GPU =                               get_hyperparam_from_config_file(config_path, 'USE_GPU')

    BATCH_SIZE_TRAIN =                      get_hyperparam_from_config_file(config_path, 'BATCH_SIZE_TRAIN')
    BATCH_SIZE_VAL =                        get_hyperparam_from_config_file(config_path, 'BATCH_SIZE_VAL')
    BATCH_SIZE_TEST =                       get_hyperparam_from_config_file(config_path, 'BATCH_SIZE_TEST')

    LEARNING_RATE =                         get_hyperparam_from_config_file(config_path, 'LEARNING_RATE')
    LEARNING_RATE /= 1e6

    TRAIN_DATA_PATH =                       get_hyperparam_from_config_file(config_path, 'TRAIN_DATA_PATH')
    VAL_DATA_PATH =                         get_hyperparam_from_config_file(config_path, 'VAL_DATA_PATH')
    TEST_DATA_PATH =                        get_hyperparam_from_config_file(config_path, 'TEST_DATA_PATH')
    DATA_PATH =                             get_hyperparam_from_config_file(config_path, 'DATA_PATH')
    ROOT_PATH =                             get_hyperparam_from_config_file(config_path, 'ROOT_PATH')

    TRAIN_IMAGES_MEAN_FILE_PATH =           get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_MEAN_FILE_PATH')
    TRAIN_IMAGES_STD_FILE_PATH  =           get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_STD_FILE_PATH')

    TRAIN_IMAGES_TOTAL_MEAN_FILE_PATH =     get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_TOTAL_MEAN_FILE_PATH')
    TRAIN_IMAGES_TOTAL_STD_FILE_PATH  =     get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_TOTAL_STD_FILE_PATH')

    MAX_TOTAL_IMAGE_NUMBER =                get_hyperparam_from_config_file(config_path, 'MAX_TOTAL_IMAGE_NUMBER')
    LOGGER_PATH =                           get_hyperparam_from_config_file(config_path, 'LOGGER_PATH')
    PCA_decomp_in_every_epochs =            get_hyperparam_from_config_file(config_path, 'PCA_decomp_in_every_epochs')
    run_id =                                get_hyperparam_from_config_file(config_path, 'run_id')

    #GENERATE_DATA_FROM_START =              get_hyperparam_from_config_file(config_path, 'GENERATE_DATA_FROM_START')
    #ONLY_GENERATE_DATA   =                  get_hyperparam_from_config_file(config_path, 'ONLY_GENERATE_DATA')

    # load train/val/test image ids and check if their number adds up to total number of images
    train_shuffled_image_ids= np.load(ROOT_PATH+"train_shuffled_image_ids.npy")
    val_shuffled_image_ids  = np.load(ROOT_PATH+"val_shuffled_image_ids.npy")
    test_shuffled_image_ids = np.load(ROOT_PATH+"test_shuffled_image_ids.npy")
    assert(set(np.concatenate((train_shuffled_image_ids,val_shuffled_image_ids,test_shuffled_image_ids))) == set(np.arange(MAX_TOTAL_IMAGE_NUMBER)))

    # calculate variance in data


    ###########################################
    # DATA PREPROCESSING - creating Transforms#
    ###########################################

    zero_mean_unit_std_transform = transforms.Compose([
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
        ]) # OUTPUT (1/2)*TANH of DNN

    # Pick one transform that is applied
    #TRANSFORM_IMG = zero_min_one_max_transform#zero_mean_unit_std_transform # zero_min_one_max_transform
    TRANSFORM_IMG = minus_one_min_one_max_transform#zero_mean_unit_std_transform # zero_min_one_max_transform

    ################
    # DATA LOADING #
    ################

    # Train Data & Train data Loader
    args_train = {'TOTAL_NUMBER_OF_IMAGES' : MAX_TOTAL_IMAGE_NUMBER, 'image_ids' : train_shuffled_image_ids}
    train_data = CustomImageDataset(args = args_train, root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=NUM_WORKERS)

    # Validation Data & Validation data Loader
    args_val = {'TOTAL_NUMBER_OF_IMAGES' : MAX_TOTAL_IMAGE_NUMBER, 'image_ids' : val_shuffled_image_ids}
    val_data = CustomImageDataset(args = args_val, root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=BATCH_SIZE_VAL, shuffle=True,  num_workers=NUM_WORKERS)

    # Test Data & Test data Loader
    args_test = {'TOTAL_NUMBER_OF_IMAGES' : MAX_TOTAL_IMAGE_NUMBER, 'image_ids' : test_shuffled_image_ids}
    test_data = CustomImageDataset(args = args_test, root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_data_loader  = torch.utils.data.DataLoader(dataset = test_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=NUM_WORKERS) 

    # class Manual_Encoder(nn.Module):
    #     def __init__(self, args_encoder, res_block_args):
    #         super(Manual_Encoder, self).__init__()
    #         self.args_encoder = args_encoder
    #         nb_layers_in_a_block = 2
    #         C_out_1_init, multiplier_value = 256//8, 2#32 #=256//8  #32#good for K=8 but for higer K (e.g. 128) you can go lower for C_out_1_init to not overfitt the data
    #         C_out_1_init, multiplier_value = args_encoder['C_out_1_init'],args_encoder['multiplier_value']
    #         k,s,p=4,2,1
    #         l=0
    #         self.sequential_convs = torch.nn.Sequential()
    #         if self.args_encoder['M'] <= 31: 
    #             self.sequential_convs.add_module(f"conv2d_{l}", nn.Conv2d(in_channels=3, out_channels=C_out_1_init, kernel_size=k, stride=s, padding=p))
    #             self.sequential_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 32
    #         if self.args_encoder['M'] <= 15:
    #             l+=1
    #             self.sequential_convs.add_module(f"conv2d_{l}", nn.Conv2d(in_channels=self.sequential_convs[0].out_channels, out_channels=multiplier_value * self.sequential_convs[0].out_channels, kernel_size=k, stride=s, padding=p))
    #             self.sequential_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 64
    #         if self.args_encoder['M'] <= 7:
    #             l+=1
    #             self.sequential_convs.add_module(f"conv2d_{l}", nn.Conv2d(in_channels=self.sequential_convs[nb_layers_in_a_block].out_channels, out_channels=multiplier_value * self.sequential_convs[nb_layers_in_a_block].out_channels, kernel_size=k, stride=s, padding=p))
    #             self.sequential_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 128
    #         if self.args_encoder['M'] <= 3:
    #             l+=1
    #             self.sequential_convs.add_module(f"conv2d_{l}", nn.Conv2d(in_channels=self.sequential_convs[2*nb_layers_in_a_block].out_channels, out_channels=multiplier_value * self.sequential_convs[2*nb_layers_in_a_block].out_channels, kernel_size=k, stride=s, padding=p))
    #             self.sequential_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 256
                
    #         res_block_args['C_in'] = self.sequential_convs[-nb_layers_in_a_block].out_channels
    #         self.residual_stack = ResidualStack(res_block_args) 
            
    #         if self.args_encoder['M'] == 1:
    #             self.avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
    #         self.channel_adjusting_conv = nn.Conv2d(in_channels=self.sequential_convs[-nb_layers_in_a_block].out_channels, out_channels=args_encoder['D'], kernel_size=1, stride=1, padding=0)
    #         #C_out = D
            
    #     def forward(self, x):
    #         x = self.sequential_convs(x)
    #         x = self.residual_stack(x)
    #         if self.args_encoder['M'] == 1:
    #             x = self.avg_pooling(x)
    #         x = self.channel_adjusting_conv(x)
    #         return x

    # class Manual_Decoder(nn.Module):
    #     def __init__(self, args_decoder, res_block_args):
    #         super(Manual_Decoder, self).__init__()
    #         self.args_decoder = args_decoder
            
    #         #C_in_1_init, divisor_value = 256, 2 ##256#good for K=8 but for higer K (e.g. 128) you can go lower for C_in_1_init to not overfitt the data
            
    #         C_in_1_init, divisor_value = args_decoder['C_in_1_init'], args_decoder['divisor_value']
            
    #         res_block_args['C_in'] = C_in_1_init
            
    #         self.channel_adjusting_conv = nn.Conv2d(in_channels=args_decoder['D'], out_channels=C_in_1_init, kernel_size=1, stride=1, padding=0)
    #         self.residual_stack = ResidualStack(res_block_args) 
    #         self.sequential_trans_convs = torch.nn.Sequential()
            
    #         nb_layers_in_a_block = 2
    #         k,s,p=4,2,1
    #         l=0
    #         if self.args_decoder['M'] <= 31: 
    #             self.sequential_trans_convs.add_module(f"trans_conv{l}", nn.ConvTranspose2d(in_channels=C_in_1_init, out_channels = C_in_1_init // divisor_value, kernel_size=k, stride=s, padding=p))
    #             self.sequential_trans_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 128
    #         if self.args_decoder['M'] <= 15:
    #             l+=1
    #             self.sequential_trans_convs.add_module(f"trans_conv{l}", nn.ConvTranspose2d(in_channels=self.sequential_trans_convs[0].out_channels, out_channels=self.sequential_trans_convs[0].out_channels // divisor_value, kernel_size=k, stride=s, padding=p))
    #             self.sequential_trans_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 64
    #         if self.args_decoder['M'] <= 7:
    #             l+=1
    #             self.sequential_trans_convs.add_module(f"trans_conv{l}", nn.ConvTranspose2d(in_channels=self.sequential_trans_convs[nb_layers_in_a_block].out_channels, out_channels=self.sequential_trans_convs[nb_layers_in_a_block].out_channels // divisor_value, kernel_size=k, stride=s, padding=p))
    #             self.sequential_trans_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 32
    #         if self.args_decoder['M'] <= 3:
    #             l+=1
    #             self.sequential_trans_convs.add_module(f"trans_conv{l}", nn.ConvTranspose2d(in_channels=self.sequential_trans_convs[2*nb_layers_in_a_block].out_channels, out_channels=self.sequential_trans_convs[2*nb_layers_in_a_block].out_channels // divisor_value, kernel_size=k, stride=s, padding=p))
    #             self.sequential_trans_convs.add_module(f"ReLU_{l}", nn.ReLU(True))
    #             #C_out = 16
                
    #         self.output_conv_layer =  nn.Conv2d(in_channels=self.sequential_trans_convs[-nb_layers_in_a_block].out_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
            
    #     def forward(self, x):
    #         x = self.channel_adjusting_conv(x)
    #         x = self.residual_stack(x)
    #         x = self.sequential_trans_convs(x)
    #         x = self.output_conv_layer(x)
    #         return x
        
    # class VQ_VAE(nn.Module):
    #     def __init__(self, args_encoder, args_VQ, args_decoder, res_block_args_encoder, res_block_args_decoder):
    #         super(VQ_VAE, self).__init__()
            
    #         ######################
    #         # Model Constructors #
    #         ######################
    #         self.C_in, self.H_in, self.W_in = args_encoder['C_in'], args_encoder['H_in'], args_encoder['W_in']
    #         self.args_encoder=args_encoder
    #         self.args_VQ=args_VQ
    #         self.args_decoder=args_decoder
    #         self.res_block_args_encoder = res_block_args_encoder
    #         self.res_block_args_decoder = res_block_args_decoder
            
    #         self.train_with_quantization = args_VQ['train_with_quantization']
            
    #         #self.encoder =  Encoder(self.args_encoder, self.res_block_args_encoder)
    #         self.encoder =  Manual_Encoder(self.args_encoder, self.res_block_args_encoder)
    #         if self.train_with_quantization:
    #             self.VQ      =  VectorQuantizer(args_VQ)
    #         #self.decoder =  Decoder(self.args_decoder, self.res_block_args_decoder)
    #         self.decoder =  Manual_Decoder(self.args_decoder, self.res_block_args_decoder)
            

    #     def forward(self, x):                       #torch.Size([128, 3, 64, 64])
    #         Ze = self.encoder(x)                    #torch.Size([128, 64, 16, 16])
            
    #         e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = None, None, None, None, None
    #         if self.train_with_quantization:
    #             e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = self.VQ(Ze)   #torch.Size([128, 64, 16, 16])
    #         else:
    #             e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy =  0, Ze, 0, 0, 0
    #         x_recon = self.decoder(Zq)              #torch.Size([128, 3, 64, 64])
    #         return e_and_q_latent_loss, x_recon, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy

    #########################
    # VQ-VAE model creation #
    #########################
    # VQ args
    args_VQ = {}
    for arg_VQ in ['max_channel_number', 'train_with_quantization', 'D', 'K', 'beta', 'M', 'use_EMA', 'gamma', 'requires_normalization_with_sphere_projection']:
        args_VQ[arg_VQ] = get_hyperparam_from_config_file(config_path, arg_VQ)
    K,D,M = args_VQ['K'],args_VQ['D'],args_VQ['M']
    # Encoder Residual Block arguments
    res_block_args_encoder = {'block_size' : get_hyperparam_from_config_file(config_path, 'res_block_size'), 'C_mid' : get_hyperparam_from_config_file(config_path, 'res_blocks_channel_number_in_hidden_layers')}
    # Decoder Residual Block arguments
    res_block_args_decoder = {'block_size' : get_hyperparam_from_config_file(config_path, 'res_block_size'), 'C_mid' : get_hyperparam_from_config_file(config_path, 'res_blocks_channel_number_in_hidden_layers')}
    # Encoder and Decoder args
    args_encoder = {'M' : args_VQ['M'], 'D' : args_VQ['D'], 'C_in' : C, 'H_in' : H, 'W_in' : W , 'use_BN' : get_hyperparam_from_config_file(config_path, 'use_BN')}
    args_decoder = {'M' : args_VQ['M'], 'D' : args_VQ['D'], 'use_BN' : get_hyperparam_from_config_file(config_path, 'use_BN')}

    # channel hyper params
    nb_of_conv2d_stride_2_layers = {31 : 1, 15: 2, 7: 3, 3: 4, 1:5, 0:6}

    args_decoder['C_in_1_init'], args_decoder['divisor_value'] = 256, get_hyperparam_from_config_file(config_path, 'divisor_value')
    args_encoder['C_out_1_init'], args_encoder['multiplier_value'] = args_decoder['C_in_1_init'] // (args_decoder['divisor_value']**(nb_of_conv2d_stride_2_layers[M] - 1)), args_decoder['divisor_value']

    # VQ VAE model
    model = VQ_VAE(args_encoder, args_VQ, args_decoder, res_block_args_encoder, res_block_args_decoder)
    #print(model)
    count_parameters(model, path_to_write = LOGGER_PATH)
    report_cuda_memory_status()


    #model = vq_vae_implemented_model

    # create a trainer init arguments
    training_args = {}
    training_args['NUM_EPOCHS']         = NUM_EPOCHS
    training_args['loss_fn']            = nn.MSELoss()
    training_args['device']             = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu") 
    training_args['model']              = model #vq_vae_implemented_model
    training_args['model_name']         = 'VQ_VAE'
    training_args['loaders']            = {'train' : train_data_loader, 'val' : val_data_loader, 'test' : test_data_loader}
    training_args['optimizer_settings'] = {'optimization_algorithm':'Adam','lr':LEARNING_RATE}
    training_args['logger_path'] = LOGGER_PATH
    # if PCA_decomp_in_every_epochs True then it considerably (from 50mins to 70 mins, i.e. 40%!) slows down the training loop!!!
    training_args['PCA_decomp_in_every_epochs'] = True
    # .item() because it is one element np.array; and we square it because we want variance and not the standard deviation
    training_args['train_data_variance'] = np.load(TRAIN_IMAGES_TOTAL_STD_FILE_PATH).item() **2
    # because we divided the chanells with TRANSFORM_IMG.std[0] we have to correct the total training data variance for that in other words training_args['train_data_variance'] was VAR[X] but because we did linear transform TRANSFORM_IMG so that X -> (X - MEAN_TRANSFORM_IMG) / STD_TRANSFORM_IMG we need to adjust the total variance of the data from VAR[X] -> VAR[(X - MEAN_TRANSFORM_IMG) / STD_TRANSFORM_IMG] = VAR[X] / STD_TRANSFORM_IMG**2 and that is precicely what we are doing here
    training_args['train_data_variance'] /= (TRANSFORM_IMG.transforms[0].std[0]**2)
    # training_args['train_data_variance']=1.
    print(f"Inverse of training data variance term is equal to =  {1. / training_args['train_data_variance']:.1f}")




    compressed_number_of_bits_per_image = int(np.ceil(np.log2(model.args_VQ['K']))) * (model.args_VQ['M']+1) ** 2

    trainer_folder_path = ROOT_PATH + \
                        str(run_id).zfill(3) + "_" + training_args['model_name'] + \
                        '_K_' + str(model.args_VQ['K']) + \
                        '_D_' + str(model.args_VQ['D']) + \
                        '_M_' + str(model.args_VQ['M']) + \
                        '_bits_' + str(compressed_number_of_bits_per_image)
    training_args['main_folder_path']   = trainer_folder_path

    if not(os.path.exists(trainer_folder_path)):
        os.system(f'mkdir {trainer_folder_path}')

    # create a trainer object
    trainer = Model_Trainer(args=training_args)

    if not USE_PRETRAINED_MODEL:
        current_time_str = time.strftime("%H:%M:%S %d.%m.%Y", time.gmtime(time.time()))
        log_str = f"[{current_time_str}] {run_id}) Started running for K = {K} & D = {D} & M = {M} & beta = {model.args_VQ['beta']} & max_channel_number = {args_decoder['C_in_1_init']} (i.e. bits = {compressed_number_of_bits_per_image}) change_channel_size_across_layers by factor = {args_decoder['divisor_value']}"
        
        with open(LOGGER_PATH, 'a') as f:
            # get current time in the format hh:mm:ss DD.MM.YYYY
            f.write(f"****************************************************************************************************************\n\n")
            f.write(f"----- {current_time_str} BEGIN RUN -----\n\n")
            f.write(f"Started {log_str}:\n\n--------------------------------------------------- \n\n")

        START_TIME = time.time()
        #train
        trainer.train()
        #train
        TOTAL_TRAINING_TIME = int(np.floor(time.time() - START_TIME))
        minutes, seconds = divmod(TOTAL_TRAINING_TIME, 60)
        hours, minutes = divmod(minutes, 60)
        TOTAL_TRAINING_TIME = f"{hours}:{minutes}:{seconds} h/m/s"
        
        # get current time in the format hh:mm:ss DD.MM.YYYY
        current_time_str = time.strftime("%H:%M:%S %d.%m.%Y", time.gmtime(time.time()))
        with open(LOGGER_PATH, 'a') as f:
            f.write(f"Finished {log_str}:\nTotal training time is = {TOTAL_TRAINING_TIME}. \n\n--------------------------------------------------- \n\n")
            f.write(f"----- {current_time_str} END RUN -----\n\n****************************************************************************************************************\n\n")

    if USE_PRETRAINED_MODEL:
        ##########################    
        # Use a pretrained model #
        ##########################
        trainer.epoch_ids_PCA = list(range( int(0.05*trainer.NUM_EPOCHS), trainer.NUM_EPOCHS + 1, int(0.05*trainer.NUM_EPOCHS)))
        
        if run_id == 600:
            current_time_str = "2023_01_18_01_34_27" # "YYYY_MM_DD_HH_MM_SS" 
        else:
            assert(False)
        
        # load model that was trained at newly given current_time_str 
        trainer.load_model(current_time_str = current_time_str, autoencoder_config_params_wrapped_sorted= None)

        # load avg. training loss of the training proceedure for a model that was trained at newly given current_time_str 
        trainer.train_loss_avg = np.load(trainer.main_folder_path + '/' + trainer.model_name + '_train_loss_avg_' + trainer.current_time_str + '.npy')
        
        # load avg. validation loss of the validating proceedure for a model that was trained at newly given current_time_str 
        trainer.val_loss_avg = np.load(trainer.main_folder_path + '/' + trainer.model_name + '_val_loss_avg_' + trainer.current_time_str + '.npy')
        
        # Load individual loss terms for both training and validation datasets
        trainer.usage_of_multiple_terms_loss_function = True
        # Training Loss data per term
        trainer.train_multiple_losses_avg = {}
        
        # Validation Loss data per term
        trainer.val_multiple_losses_avg = {}
        
        # Training Loss data per term file path
        trainer.train_multiple_losses_avg_path = {}
        
        # Validation Loss data per term file path
        trainer.val_multiple_losses_avg_path = {}
        
        for loss_term in ['reconstruction_loss','commitment_loss', 'VQ_codebook_loss']:
            # Training Loss data per term file path
            trainer.train_multiple_losses_avg_path[loss_term] = trainer.main_folder_path + '/' + trainer.model_name + '_train_multiple_losses_avg_' + loss_term + '_'  + trainer.current_time_str + '.npy'
            
            # Validation Loss data per term file path
            trainer.val_multiple_losses_avg_path[loss_term]   = trainer.main_folder_path + '/' + trainer.model_name + '_val_multiple_losses_avg_' + loss_term + '_'  + trainer.current_time_str + '.npy'
            
            # Training Loss data per term
            trainer.train_multiple_losses_avg[loss_term] = np.load(trainer.train_multiple_losses_avg_path[loss_term])
            
            # Validation Loss data per term
            trainer.val_multiple_losses_avg[loss_term]   = np.load(trainer.val_multiple_losses_avg_path[loss_term])
        
        # Load Perplexity over epochs during training
        trainer.train_metrics, trainer.val_metrics = {}, {}    
        trainer.train_metrics_perplexity_path = trainer.main_folder_path + '/' + trainer.model_name + '_train_perplexity_' + trainer.current_time_str + '.npy'
        trainer.val_metrics_perplexity_path = trainer.main_folder_path + '/' + trainer.model_name + '_val_perplexity_' + trainer.current_time_str + '.npy'
        trainer.train_metrics['perplexity'] = np.load(trainer.train_metrics_perplexity_path)
        trainer.val_metrics['perplexity'] = np.load(trainer.val_metrics_perplexity_path)
        
    #########################
    ### Testing the model ###
    #########################
    loss_fn = trainer.loss_fn
    loss_fn.to(trainer.device)
    trainer.test() 

    #############################################################################
    # Plot train and validation avergae loss across mini-batch across epochs #
    # Plot Test Loss for every sample in the Test set #
    #############################################################################
    trainer.plot()

    ############################################################
    # Plot top-N worst reconstructed test images
    # [with their original test images side by side
    # and rank them from worst (highest reconstruction loss value)
    # to best reconstructed test image]
    ############################################################
    TOP_WORST_RECONSTRUCTED_TEST_IMAGES = 50
    trainer.get_worst_test_samples(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)
    trainer.model.eval()
    visualise_output(images             = trainer.worst_top_images, 
                    model              = trainer.model,
                    compose_transforms = TRANSFORM_IMG,
                    imgs_ids           = trainer.worst_imgs_ids,
                    imgs_losses        = trainer.worst_imgs_losses,
                    savefig_path       = trainer.main_folder_path + '/WORST_RECONSTRUCTED_TEST_IMAGES.png',
                    device = trainer.device)

    ############################################################
    # Plot top-N best reconstructed test images
    # [with their original test images side by side
    # and rank them from best (lowest reconstruction loss value)
    # to worst reconstructed test image]
    ############################################################
    TOP_BEST_RECONSTRUCTED_TEST_IMAGES = 50
    trainer.get_best_test_samples(TOP_BEST_RECONSTRUCTED_TEST_IMAGES)
    trainer.model.eval()
    visualise_output(images             = trainer.best_top_images, 
                    model              = trainer.model,
                    compose_transforms = TRANSFORM_IMG,
                    imgs_ids           = trainer.best_imgs_ids,
                    imgs_losses        = trainer.best_imgs_losses,
                    savefig_path       = trainer.main_folder_path + '/BEST_RECONSTRUCTED_TEST_IMAGES.png',
                    device = trainer.device)

    ################################################
    ### Training & Validation metrics visualized ###
    ################################################
    if not USE_PRETRAINED_MODEL:
        trainer.plot_perlexity()

    #####################################################################
    ### Codebook (a matrix of codewords) and Tokens Z_Q visualization ###
    #####################################################################
    #trainer.codebook_visualization()#nije bitno
    #trainer.plot_codebook_PCA()
    trainer.visualize_discrete_codes(compose_transforms = TRANSFORM_IMG, dataset_str = 'test')
