import time
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from models import Model_Trainer, CustomImageDataset
from VQ_VAE import VQ_VAE
from helper_functions import get_hyperparam_from_config_file, count_parameters, report_cuda_memory_status, visualise_output

def inference(config_path = "/home/novakovm/iris/MILOS/toy_shapes_config.yaml"):
    ROOT_PATH =     get_hyperparam_from_config_file(config_path, 'ROOT_PATH')  
    #current_working_absoulte_path = '/home/novakovm/iris/MILOS'
    #os.chdir(ROOT_PATH)

    # number of images for training and testing datasets
    MAX_TOTAL_IMAGE_NUMBER  = get_hyperparam_from_config_file(config_path, 'MAX_TOTAL_IMAGE_NUMBER')
    #TEST_MAX_TOTAL_IMAGE_NUMBER = get_hyperparam_from_config_file(config_path, 'TEST_MAX_TOTAL_IMAGE_NUMBER')

    # load train/val/test dataset percentage take adds up to 100 (percent)
    train_dataset_percentage= get_hyperparam_from_config_file(config_path, 'train_dataset_percentage')
    val_dataset_percentage  = get_hyperparam_from_config_file(config_path, 'val_dataset_percentage')
    test_dataset_percentage = get_hyperparam_from_config_file(config_path, 'test_dataset_percentage')
    assert(100 ==train_dataset_percentage+val_dataset_percentage+test_dataset_percentage)

    # load train/val/test image ids and check if their number adds up to total number of images
    train_shuffled_image_ids= np.load(ROOT_PATH+'train_shuffled_image_ids.npy')
    val_shuffled_image_ids  = np.load(ROOT_PATH+'val_shuffled_image_ids.npy')
    test_shuffled_image_ids = np.load(ROOT_PATH+'test_shuffled_image_ids.npy')

    all_nonshuffled_image_ids = np.load(ROOT_PATH+'all_nonshuffled_image_ids.npy')
    assert(set(np.concatenate((train_shuffled_image_ids,val_shuffled_image_ids,test_shuffled_image_ids))) == set(all_nonshuffled_image_ids))

    # init config args for train/val/test loaders
    args_train, args_val, args_test = {}, {}, {}
    args_train['TOTAL_NUMBER_OF_IMAGES'], args_train['image_ids']= MAX_TOTAL_IMAGE_NUMBER, train_shuffled_image_ids
    args_val['TOTAL_NUMBER_OF_IMAGES'],   args_val['image_ids']  = MAX_TOTAL_IMAGE_NUMBER, val_shuffled_image_ids
    args_test['TOTAL_NUMBER_OF_IMAGES'], args_test['image_ids']  = MAX_TOTAL_IMAGE_NUMBER, test_shuffled_image_ids

    # Height, Width, Channel number
    H=                                      get_hyperparam_from_config_file(config_path, 'H')
    W=                                      get_hyperparam_from_config_file(config_path, 'W')
    C=                                      get_hyperparam_from_config_file(config_path, 'C')
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
    TRAIN_IMAGES_MEAN_FILE_PATH =           get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_MEAN_FILE_PATH')
    TRAIN_IMAGES_STD_FILE_PATH  =           get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_STD_FILE_PATH')
    TRAIN_IMAGES_TOTAL_MEAN_FILE_PATH =     get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_TOTAL_MEAN_FILE_PATH')
    TRAIN_IMAGES_TOTAL_STD_FILE_PATH  =     get_hyperparam_from_config_file(config_path, 'TRAIN_IMAGES_TOTAL_STD_FILE_PATH')
    LOGGER_PATH =                           get_hyperparam_from_config_file(config_path, 'LOGGER_PATH')
    PCA_decomp_in_every_epochs =            get_hyperparam_from_config_file(config_path, 'PCA_decomp_in_every_epochs')
    run_id =                                get_hyperparam_from_config_file(config_path, 'run_id')
    USE_PRETRAINED_MODEL_run_id =           get_hyperparam_from_config_file(config_path, 'USE_PRETRAINED_MODEL_run_id')
    USE_PRETRAINED_MODEL_current_time_str = get_hyperparam_from_config_file(config_path, 'USE_PRETRAINED_MODEL_current_time_str')
    
    if USE_PRETRAINED_MODEL:
        run_id = USE_PRETRAINED_MODEL_run_id

    RUN_MODEL_TESTING =                     get_hyperparam_from_config_file(config_path, 'RUN_MODEL_TESTING')
    RUN_TRAIN_VAL_AVG_LOSS_OVER_EPOCHS=     get_hyperparam_from_config_file(config_path, 'RUN_TRAIN_VAL_AVG_LOSS_OVER_EPOCHS')
    RUN_SCATTER_FOR_SPECIFIC_CLASSES =      get_hyperparam_from_config_file(config_path, 'RUN_SCATTER_FOR_SPECIFIC_CLASSES')
    RUN_TOP_WORST_RECONSTRUCTED_TEST_IMAGES=get_hyperparam_from_config_file(config_path, 'RUN_TOP_WORST_RECONSTRUCTED_TEST_IMAGES')
    RUN_TOP_BEST_RECONSTRUCTED_TEST_IMAGES= get_hyperparam_from_config_file(config_path, 'RUN_TOP_BEST_RECONSTRUCTED_TEST_IMAGES')
    RUN_MODEL_VIZUALIZATION =               get_hyperparam_from_config_file(config_path, 'RUN_MODEL_VIZUALIZATION')
    RUN_CODEBOOK_VISUALIZATION =            get_hyperparam_from_config_file(config_path, 'RUN_CODEBOOK_VISUALIZATION')
    RUN_PCA_ON_CODEBOOK =                   get_hyperparam_from_config_file(config_path, 'RUN_PCA_ON_CODEBOOK')
    RUN_TOKEN_VIZUALIZATION =               get_hyperparam_from_config_file(config_path, 'RUN_TOKEN_VIZUALIZATION')
    SAVE_RESULTS_TO_CSV =                   get_hyperparam_from_config_file(config_path, 'SAVE_RESULTS_TO_CSV')
    VISUALIZE_THE_CHANGE_OF_CODEBOOK_TOKENS=get_hyperparam_from_config_file(config_path, 'VISUALIZE_THE_CHANGE_OF_CODEBOOK_TOKENS')
    
    SET_VAR_TO_ONE  =                      get_hyperparam_from_config_file(config_path, 'SET_VAR_TO_ONE')
      
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
        ]) # OUTPUT (1/2)*TANH of DNN

    # Pick one transform that is applied
    TRANSFORM_IMG = zero_min_one_max_transform#zero_mean_unit_std_transform # zero_min_one_max_transform
    TRANSFORM_IMG = minus_one_min_one_max_transform#zero_mean_unit_std_transform # zero_min_one_max_transform
    # Train Data & Train data Loader
    train_data = CustomImageDataset(args = args_train, root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=NUM_WORKERS)
    # Validation Data & Validation data Loader
    val_data = CustomImageDataset(args = args_val, root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=BATCH_SIZE_VAL, shuffle=True,  num_workers=NUM_WORKERS)
    # Test Data & Test data Loader
    test_data = CustomImageDataset(args = args_test, root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_data_loader  = torch.utils.data.DataLoader(dataset = test_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=NUM_WORKERS) 

    args_VQ = {}
    for arg_VQ in ['max_channel_number', 'train_with_quantization', 'D', 'K', 'beta', 'M', 'use_EMA', 'gamma', 'requires_normalization_with_sphere_projection']:
        args_VQ[arg_VQ] = get_hyperparam_from_config_file(config_path, arg_VQ)
    K,D,M = args_VQ['K'],args_VQ['D'],args_VQ['M']
    beta, max_channel_number = args_VQ['beta'], args_VQ['max_channel_number']

    # Encoder Residual Block arguments
    res_block_args_encoder = {'block_size' : get_hyperparam_from_config_file(config_path, 'res_block_size'), 
                              'C_mid' : get_hyperparam_from_config_file(config_path, 'res_blocks_channel_number_in_hidden_layers'),
                              'res_block_use_BN' : get_hyperparam_from_config_file(config_path, 'res_block_use_BN'),
                              'res_block_use_bias' : get_hyperparam_from_config_file(config_path, 'res_block_use_bias')
                              }
    # Decoder Residual Block arguments
    res_block_args_decoder = res_block_args_encoder.copy()
    # Encoder and Decoder args
    args_encoder = {'M' : args_VQ['M'], 'D' : args_VQ['D'], 'C_in' : C, 'H_in' : H, 'W_in' : W , 'use_BN' : get_hyperparam_from_config_file(config_path, 'use_BN')}
    args_decoder = {'M' : args_VQ['M'], 'D' : args_VQ['D'], 'use_BN' : get_hyperparam_from_config_file(config_path, 'use_BN')}

    # channel hyper params
    nb_of_conv2d_stride_2_layers = {31 : 1, 
                                    15: 2,
                                    7: 3,
                                    3: 4,
                                    1:5,
                                    0:6}
    args_decoder['C_in_1_init'], args_decoder['divisor_value'] = args_VQ['max_channel_number'],  get_hyperparam_from_config_file(config_path, 'divisor_value')
    args_encoder['C_out_1_init'], args_encoder['multiplier_value'] = args_decoder['C_in_1_init'] // (args_decoder['divisor_value']**(nb_of_conv2d_stride_2_layers[M] - 1)), args_decoder['divisor_value']
    change_channel_size_across_layers = (1 != args_decoder['divisor_value'])

    # VQ VAE model
    model = VQ_VAE(args_encoder, args_VQ, args_decoder, res_block_args_encoder, res_block_args_decoder)
    count_parameters(model, path_to_write = LOGGER_PATH)
    report_cuda_memory_status()

    # set the training parameters
    # create a trainer init arguments
    training_args = {}
    training_args['NUM_EPOCHS']         = NUM_EPOCHS
    training_args['loss_fn']            = nn.MSELoss()
    training_args['device']             = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu") 
    training_args['model']              = model
    training_args['model_name']         = get_hyperparam_from_config_file(config_path, 'model_name')
    training_args['loaders']            = {'train' : train_data_loader, 'val' : val_data_loader, 'test' : test_data_loader}
    training_args['optimizer_settings'] = {'optimization_algorithm':'Adam','lr':LEARNING_RATE} #torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-5)
    training_args['logger_path']        = LOGGER_PATH
    training_args['config_path']        = config_path
    # if PCA_decomp_in_every_epochs True then it considerably (from 50mins to 70 mins, i.e. 40%!) slows down the training loop!!!
    training_args['PCA_decomp_in_every_epochs'] = PCA_decomp_in_every_epochs
    # .item() because it is one element np.array; and we square it because we want variance and not the standard deviation
    training_args['train_data_variance'] = np.load(TRAIN_IMAGES_TOTAL_STD_FILE_PATH).item() **2
    # because we divided the chanells with TRANSFORM_IMG.std[0] we have to correct the total training data variance for that in other words training_args['train_data_variance'] was VAR[X] but because we did linear transform TRANSFORM_IMG so that X -> (X - MEAN_TRANSFORM_IMG) / STD_TRANSFORM_IMG we need to adjust the total variance of the data from VAR[X] -> VAR[(X - MEAN_TRANSFORM_IMG) / STD_TRANSFORM_IMG] = VAR[X] / STD_TRANSFORM_IMG**2 and that is precicely what we are doing here
    training_args['train_data_variance'] /= (TRANSFORM_IMG.transforms[0].std[0]**2)
    if SET_VAR_TO_ONE:
        training_args['train_data_variance']=1.
    print(f"Inverse of training data variance term is equal to =  {1. / training_args['train_data_variance']:.1f}")

    compressed_number_of_bits_per_image = int(np.ceil(np.log2(K))) * (M+1) ** 2
    trainer_folder_path = ROOT_PATH + \
                        str(run_id).zfill(3) + "_" + training_args['model_name'] + \
                        '_K_' + str(model.args_VQ['K']) + \
                        '_D_' + str(model.args_VQ['D']) + \
                        '_M_' + str(model.args_VQ['M']) + \
                        '_bits_' + str(compressed_number_of_bits_per_image) + "/"
    training_args['main_folder_path']   = trainer_folder_path

    if not(os.path.exists(trainer_folder_path)):
        os.system(f'mkdir {trainer_folder_path}')

    # create a trainer object
    trainer = Model_Trainer(args=training_args)
    trainer.main_folder_path   =  trainer_folder_path

    if not USE_PRETRAINED_MODEL:
        # start the training and validation procedure
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

    else:
        ##########################    
        # Use a pretrained model #
        ##########################
        trainer.epoch_ids_PCA = list(range( int(0.05*trainer.NUM_EPOCHS), trainer.NUM_EPOCHS + 1, int(0.05*trainer.NUM_EPOCHS)))
        
        run_id = USE_PRETRAINED_MODEL_run_id
        current_time_str = USE_PRETRAINED_MODEL_current_time_str

        # load model that was trained at newly given current_time_str 
        trainer.load_model(current_time_str = current_time_str)

        # load avg. training loss of the training proceedure for a model that was trained at newly given current_time_str 
        trainer.train_loss_avg = np.load(trainer.main_folder_path + trainer.model_name + '_train_loss_avg_' + trainer.current_time_str + '.npy')
        
        # load avg. validation loss of the validating proceedure for a model that was trained at newly given current_time_str 
        trainer.val_loss_avg = np.load(trainer.main_folder_path + trainer.model_name + '_val_loss_avg_' + trainer.current_time_str + '.npy')
        
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
            trainer.train_multiple_losses_avg_path[loss_term] = trainer.main_folder_path + trainer.model_name + '_train_multiple_losses_avg_' + loss_term + '_'  + trainer.current_time_str + '.npy'
            
            # Validation Loss data per term file path
            trainer.val_multiple_losses_avg_path[loss_term]   = trainer.main_folder_path + trainer.model_name + '_val_multiple_losses_avg_' + loss_term + '_'  + trainer.current_time_str + '.npy'
            
            # Training Loss data per term
            trainer.train_multiple_losses_avg[loss_term] = np.load(trainer.train_multiple_losses_avg_path[loss_term])
            
            # Validation Loss data per term
            trainer.val_multiple_losses_avg[loss_term]   = np.load(trainer.val_multiple_losses_avg_path[loss_term])
        
        # Load Perplexity over epochs during training
        trainer.train_metrics, trainer.val_metrics = {}, {}    
        trainer.train_metrics_perplexity_path = trainer.main_folder_path  + trainer.model_name + '_train_perplexity_' + trainer.current_time_str + '.npy'
        trainer.val_metrics_perplexity_path = trainer.main_folder_path  + trainer.model_name + '_val_perplexity_' + trainer.current_time_str + '.npy'
        trainer.train_metrics['perplexity'] = np.load(trainer.train_metrics_perplexity_path)
        trainer.val_metrics['perplexity'] = np.load(trainer.val_metrics_perplexity_path)
        
        trainer.min_train_loss_path = trainer.main_folder_path + trainer.model_name + '_min_train_loss_' + trainer.current_time_str + '.npy'
        trainer.min_val_loss_path = trainer.main_folder_path + trainer.model_name + '_min_val_loss_' + trainer.current_time_str + '.npy'
        #uncomment this when you run the training again
        #trainer.min_train_loss = np.load(trainer.min_train_loss_path)
        #trainer.min_val_loss = np.load(trainer.min_val_loss_path)
        
    #########################
    ### Testing the model ###
    #########################
    if RUN_MODEL_TESTING:
        loss_fn = trainer.loss_fn
        loss_fn.to(trainer.device)
        trainer.test() 

    #############################################################################
    # Plot train and validation avergae loss across mini-batch across epochs #
    # Plot Test Loss for every sample in the Test set #
    #############################################################################
    if RUN_TRAIN_VAL_AVG_LOSS_OVER_EPOCHS:
        trainer.plot()

    #############################################################################
    # Plot test images reconstruction losses
    # And for "labels" use different shape features to see  
    # which shape features did autoencoder learned the best/worst
    # (i.e. what is the easiest/hardest to learn from persepctive of autoencoder)
    #############################################################################
    if RUN_SCATTER_FOR_SPECIFIC_CLASSES:
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

    ############################################################
    # Plot top-N worst reconstructed test images
    # [with their original test images side by side
    # and rank them from worst (highest reconstruction loss value)
    # to best reconstructed test image]
    ############################################################
    if RUN_TOP_WORST_RECONSTRUCTED_TEST_IMAGES:
        TOP_WORST_RECONSTRUCTED_TEST_IMAGES = 50
        trainer.get_worst_test_samples(TOP_WORST_RECONSTRUCTED_TEST_IMAGES)
        trainer.model.eval()
        visualise_output(images             = trainer.worst_top_images, 
                        model              = trainer.model,
                        compose_transforms = TRANSFORM_IMG,
                        imgs_ids           = trainer.worst_imgs_ids,
                        imgs_losses        = trainer.worst_imgs_losses,
                        savefig_path       = trainer.main_folder_path + 'WORST_RECONSTRUCTED_TEST_IMAGES.png',
                        device = trainer.device)

    ############################################################
    # Plot top-N best reconstructed test images
    # [with their original test images side by side
    # and rank them from best (lowest reconstruction loss value)
    # to worst reconstructed test image]
    ############################################################
    if RUN_TOP_BEST_RECONSTRUCTED_TEST_IMAGES:
        TOP_BEST_RECONSTRUCTED_TEST_IMAGES = 50
        trainer.get_best_test_samples(TOP_BEST_RECONSTRUCTED_TEST_IMAGES)
        trainer.model.eval()
        visualise_output(images             = trainer.best_top_images, 
                        model              = trainer.model,
                        compose_transforms = TRANSFORM_IMG,
                        imgs_ids           = trainer.best_imgs_ids,
                        imgs_losses        = trainer.best_imgs_losses,
                        savefig_path       = trainer.main_folder_path + 'BEST_RECONSTRUCTED_TEST_IMAGES.png',
                        device = trainer.device)


    ###################################
    ### Graph of a model visualized ###
    ###################################
    if RUN_MODEL_VIZUALIZATION:
        trainer.visualize_model_as_graph_image()

    ################################################
    ### Training & Validation metrics visualized ###
    ################################################
    if not USE_PRETRAINED_MODEL:
        trainer.plot_perlexity()

    #####################################################################
    ### Codebook (a matrix of codewords) and Tokens Z_Q visualization ###
    #####################################################################
    if RUN_CODEBOOK_VISUALIZATION:
        trainer.codebook_visualization()

    if RUN_PCA_ON_CODEBOOK:
        trainer.plot_codebook_PCA()

    if RUN_TOKEN_VIZUALIZATION:
        trainer.visualize_discrete_codes(compose_transforms = TRANSFORM_IMG, dataset_str = 'test')

    ###################################################################################
    ### Logging of Results (especially useful when training large number of models) ###
    ###################################################################################
    if SAVE_RESULTS_TO_CSV:
        models_param_number_fn = lambda model : sum(p.numel() for p in model.parameters() if p.requires_grad)
        year = trainer.current_time_str[0:4]
        month = trainer.current_time_str[5:7]
        day = trainer.current_time_str[8:10]

        hour = trainer.current_time_str[11:13]
        minute = trainer.current_time_str[14:16]
        second = trainer.current_time_str[17:19]
            
        results_df = pd.DataFrame({
            ### ARGS ###
            'run_id':[run_id],
            'current time':[f"{hour}:{minute}  {day}.{month}.{year}"],
            'current time folder YYYY_MM_DD_hh_mm_ss':[trainer.current_time_str],
            'K':[K],
            'log2(K)':[int(np.log2(K))],
            'D':[D],
            'M':[M],
            'bits':[compressed_number_of_bits_per_image],
            'beta':[beta],
            'max_channel_number' : [max_channel_number],
            'change_channel_size_across_layers' : [change_channel_size_across_layers],
            'use_EMA' : [trainer.model.VQ.use_EMA],
            
            ### MODEL PARAM NUMBER ###
            'model_param_number' :  [models_param_number_fn(trainer.model)],
            'encoder_param_number' :[models_param_number_fn(trainer.model.encoder)],
            'VQ_param_number' :     [models_param_number_fn(trainer.model.VQ)],
            'decoder_param_number' :[models_param_number_fn(trainer.model.decoder)],
            
            '[percentage] encoder_param_number' :[models_param_number_fn(trainer.model.encoder) / models_param_number_fn(trainer.model) * 100],
            '[percentage] VQ_param_number' :     [models_param_number_fn(trainer.model.VQ) / models_param_number_fn(trainer.model) * 100],
            '[percentage] decoder_param_number' :[models_param_number_fn(trainer.model.decoder) / models_param_number_fn(trainer.model) * 100],
            

            ### TRAINING LOSS RESULTS ###
            'last_train_loss_value':[trainer.train_loss_avg[-1]],
            'min_train_loss_value':[trainer.min_train_loss],
            
            'last_train_loss_value:(1/var)||X - X_rec ||^2':[trainer.train_multiple_losses_avg['reconstruction_loss'][-1]],
            'last_train_loss_value: (1+beta)*||Z_e-Z_q||^2':[trainer.train_multiple_losses_avg['VQ_codebook_loss'][-1] + beta * trainer.train_multiple_losses_avg['commitment_loss'][-1]],
            'last_train_loss_value:beta*||Z_e-Z_q||^2':[beta * trainer.train_multiple_losses_avg['commitment_loss'][-1]],
            'last_train_loss_value:||Z_e-Z_q||^2':[trainer.train_multiple_losses_avg['VQ_codebook_loss'][-1]],
            
            '[percentage] last_train_loss_value:(1/var)||X - X_rec ||^2':[trainer.train_multiple_losses_avg['reconstruction_loss'][-1]/ trainer.train_loss_avg[-1] * 100],
            '[percentage] last_train_loss_value: (1+beta)*||Z_e-Z_q||^2':[(trainer.train_multiple_losses_avg['VQ_codebook_loss'][-1] + beta * trainer.train_multiple_losses_avg['commitment_loss'][-1] ) / trainer.train_loss_avg[-1] * 100],
            '[percentage] last_train_loss_value:beta*||Z_e-Z_q||^2':[beta * trainer.train_multiple_losses_avg['commitment_loss'][-1] / trainer.train_loss_avg[-1] * 100],
            '[percentage] last_train_loss_value:||Z_e-Z_q||^2':[trainer.train_multiple_losses_avg['VQ_codebook_loss'][-1]/ trainer.train_loss_avg[-1] * 100],
            
            ### VALIDATION LOSS RESULTS ###
            'last_val_loss_value':[trainer.val_loss_avg[-1]],
            'min_val_loss_value':[trainer.min_val_loss],
            
            'last_val_loss_value:(1/var)||X - X_rec ||^2':[trainer.val_multiple_losses_avg['reconstruction_loss'][-1]],
            'last_val_loss_value: (1+beta)*||Z_e-Z_q||^2':[trainer.val_multiple_losses_avg['VQ_codebook_loss'][-1] + beta * trainer.val_multiple_losses_avg['commitment_loss'][-1]],
            'last_val_loss_value:beta*||Z_e-Z_q||^2':[beta * trainer.val_multiple_losses_avg['commitment_loss'][-1]],
            'last_val_loss_value:||Z_e-Z_q||^2':[trainer.val_multiple_losses_avg['VQ_codebook_loss'][-1]],
            
            '[percentage] last_val_loss_value:(1/var)||X - X_rec ||^2':[trainer.val_multiple_losses_avg['reconstruction_loss'][-1]/ trainer.val_loss_avg[-1] * 100],
            '[percentage] last_val_loss_value: (1+beta)*||Z_e-Z_q||^2':[(trainer.val_multiple_losses_avg['VQ_codebook_loss'][-1] + beta * trainer.val_multiple_losses_avg['commitment_loss'][-1] ) / trainer.val_loss_avg[-1] * 100],
            '[percentage] last_val_loss_value:beta*||Z_e-Z_q||^2':[beta * trainer.val_multiple_losses_avg['commitment_loss'][-1] / trainer.val_loss_avg[-1] * 100],
            '[percentage] last_val_loss_value:||Z_e-Z_q||^2':[trainer.val_multiple_losses_avg['VQ_codebook_loss'][-1]/ trainer.val_loss_avg[-1] * 100],
            
            ### TEST LOSS RESULTS ###
            'mean test loss value':[trainer.test_samples_loss['total_loss'].mean()],

            'mean test loss value:(1/var)||X - X_rec ||^2':[trainer.test_samples_loss['reconstruction_loss'].mean()],
            'mean test loss value: (1+beta)*||Z_e-Z_q||^2':[trainer.test_samples_loss['VQ_codebook_loss'].mean() + beta * trainer.test_samples_loss['commitment_loss'].mean()],
            'mean test loss value:beta*||Z_e-Z_q||^2':[beta * trainer.test_samples_loss['commitment_loss'].mean()],
            'mean test loss value:||Z_e-Z_q||^2':[trainer.test_samples_loss['VQ_codebook_loss'].mean()],

            '[percentage] mean test loss value:(1/var)||X - X_rec ||^2':[trainer.test_samples_loss['reconstruction_loss'].mean() / trainer.test_samples_loss['total_loss'].mean() * 100],
            '[percentage] mean test loss value: (1+beta)*||Z_e-Z_q||^2':[(trainer.test_samples_loss['VQ_codebook_loss'].mean() + beta * trainer.test_samples_loss['commitment_loss'].mean())  / trainer.test_samples_loss['total_loss'].mean() * 100],
            '[percentage] mean test loss value:beta*||Z_e-Z_q||^2':[beta * trainer.test_samples_loss['commitment_loss'].mean() / trainer.test_samples_loss['total_loss'].mean() * 100],
            '[percentage] mean test loss value:||Z_e-Z_q||^2':[trainer.test_samples_loss['VQ_codebook_loss'].mean() / trainer.test_samples_loss['total_loss'].mean() * 100],

            ### PERPLEXITY RESULTS ###
            'last_train_perplexity':[trainer.train_metrics['perplexity'][-1]],
            'last_val_perplexity':[trainer.val_metrics['perplexity'][-1]],
            'mean test perplexity':[trainer.test_metrics['perplexity'].mean()],
            'true_perplexity':[K],
            
            ### ENTROPY RESULTS ###    
            'last_train_entropy':[np.log2(trainer.train_metrics['perplexity'][-1])],
            'last_val_entropy':[np.log2(trainer.val_metrics['perplexity'][-1])],
            'mean test entropy':[np.log2(trainer.test_metrics['perplexity'].mean())],
            'true_entropy':[np.log2(K)]
        })
        
        log_results_file_path = ROOT_PATH + 'log_results/'
        
        if not(os.path.exists(log_results_file_path)):
            os.mkdir(f'{log_results_file_path}')
        
        # saving of the log. results in a csv file
        with open(log_results_file_path + "log_results.csv", 'a') as f:
            results_df.to_csv(f, index = False, header=f.tell()==0)
            
    if VISUALIZE_THE_CHANGE_OF_CODEBOOK_TOKENS:
        # 10th WORST image
        #for i in range(5):
        i=10
        dataset_type = "test" #"toy_dataset", "test"
        image_id_in_dataset = trainer.worst_imgs_ids[i]
        image_index_in_dataset = np.where(trainer.loaders[dataset_type].dataset.image_ids == image_id_in_dataset)[0][0]
        trainer.change_one_token(dataset_type = "test", image_index_in_dataset = image_index_in_dataset)
        
        # 10th BEST image
        i=10
        dataset_type = "test" #"toy_dataset", "test"
        image_id_in_dataset = trainer.best_imgs_ids[i]
        image_index_in_dataset = np.where(trainer.loaders[dataset_type].dataset.image_ids == image_id_in_dataset)[0][0]
        trainer.change_one_token(dataset_type = "test", image_index_in_dataset = image_index_in_dataset)
        # OR PICK ANY IMAGE
        # dataset_type = "test" # ANY DATASET
        # image_id_in_dataset = trainer.best_imgs_ids[0] # ANY IMAGE ID IN THE SELECTED DATASET
        # image_index_in_dataset = np.where(trainer.loaders[dataset_type].dataset.image_ids == image_id_in_dataset)[0][0]
        # trainer.change_one_token(dataset_type = "test", image_index_in_dataset = image_index_in_dataset)
        
    return trainer