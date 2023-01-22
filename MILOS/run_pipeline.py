from helper_functions import get_hyperparam_from_config_file
from autoencoders import inference
from find_mean_std import find_mean_std
import sys
import numpy as np
import os

# config_path is either 
# "/home/novakovm/iris/MILOS/toy_shapes_config.yaml"
# or
# "/home/novakovm/iris/MILOS/crafter_config.yaml"
def run(config_path = None):
    config_path = sys.argv[1]
    
    ROOT_PATH =     get_hyperparam_from_config_file(config_path, 'ROOT_PATH')  
    GENERATE_DATA = get_hyperparam_from_config_file(config_path, 'GENERATE_DATA')
    PREPROCESS_DATA=get_hyperparam_from_config_file(config_path, 'PREPROCESS_DATA')
    INFERENCE_DATA =get_hyperparam_from_config_file(config_path, 'INFERENCE_DATA')

    if GENERATE_DATA:
        data_generator_path = get_hyperparam_from_config_file(config_path, 'data_generator_path')
        os.system(f"python {data_generator_path}")   
        
    if PREPROCESS_DATA:
        
        #########################################################
        # DATA PREPROCESSING - calculating empirical mean & std #
        #########################################################
        
        # verify that the shuffled train+val+test datasets don't leak data!
        train_shuffled_image_ids= np.load(ROOT_PATH+'train_shuffled_image_ids.npy')
        val_shuffled_image_ids  = np.load(ROOT_PATH+'val_shuffled_image_ids.npy')
        test_shuffled_image_ids = np.load(ROOT_PATH+'test_shuffled_image_ids.npy')
        all_nonshuffled_image_ids = np.load(ROOT_PATH+'all_nonshuffled_image_ids.npy')
        assert(set(np.concatenate((train_shuffled_image_ids,val_shuffled_image_ids,test_shuffled_image_ids))) == set(all_nonshuffled_image_ids))

        # setting the arguments for doing data preprocessing on the training dataset
        TRAIN_DATA_PATH =        get_hyperparam_from_config_file(config_path, 'TRAIN_DATA_PATH')
        ROOT_PATH =              get_hyperparam_from_config_file(config_path, 'ROOT_PATH')
        MAX_TOTAL_IMAGE_NUMBER = get_hyperparam_from_config_file(config_path, 'MAX_TOTAL_IMAGE_NUMBER')
        train_shuffled_image_ids = np.load(ROOT_PATH + 'train_shuffled_image_ids.npy')
        NUMPY_DOUBLE_CHECK = (100_000 >= MAX_TOTAL_IMAGE_NUMBER)
        
        # do data preprocessing on the training dataset
        find_mean_std(TOTAL_NUMBER_OF_IMAGES = MAX_TOTAL_IMAGE_NUMBER,
                      image_ids_numbers = train_shuffled_image_ids,
                      method = 'n',
                      NUMPY_DOUBLE_CHECK= NUMPY_DOUBLE_CHECK,
                      train_folder_path=TRAIN_DATA_PATH, 
                      main_folder_path=ROOT_PATH)
        
    if INFERENCE_DATA:
        return inference(config_path)
    
    return None

if __name__ == "__main__":
    run()