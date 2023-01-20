import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import get_hyperparam_from_config_file
                            
config_path = "/home/novakovm/iris/MILOS/crafter_config.yaml"

H =                     get_hyperparam_from_config_file(config_path, 'H')
W =                     get_hyperparam_from_config_file(config_path, 'W')
C =                     get_hyperparam_from_config_file(config_path, 'C')
TRAIN_DATA_PATH =       get_hyperparam_from_config_file(config_path, 'TRAIN_DATA_PATH')
VAL_DATA_PATH =         get_hyperparam_from_config_file(config_path, 'VAL_DATA_PATH')
TEST_DATA_PATH =        get_hyperparam_from_config_file(config_path, 'TEST_DATA_PATH')
DATA_PATH =             get_hyperparam_from_config_file(config_path, 'DATA_PATH')
ROOT_PATH =             get_hyperparam_from_config_file(config_path, 'ROOT_PATH')
LOGGER_PATH =           get_hyperparam_from_config_file(config_path, 'LOGGER_PATH')
MAX_TOTAL_IMAGE_NUMBER =get_hyperparam_from_config_file(config_path, 'MAX_TOTAL_IMAGE_NUMBER')
SEED=                   get_hyperparam_from_config_file(config_path, 'SEED')
# load train/val/test dataset percentage take adds up to 100 (percent)
train_dataset_percentage= get_hyperparam_from_config_file(config_path, 'train_dataset_percentage')
val_dataset_percentage  = get_hyperparam_from_config_file(config_path, 'val_dataset_percentage')
test_dataset_percentage = get_hyperparam_from_config_file(config_path, 'test_dataset_percentage')
assert(100 == train_dataset_percentage + val_dataset_percentage + test_dataset_percentage)

ELOI_DATA_PATH = "/data/alonsoel/workspace/data/crafter/dataset_1M_steps_heuristic_crafter/"

MAX_NUMBER_OF_PT_FILES = 5727 # read this manually in the ELOI_DATA_PATH folder

if not os.path.exists(ROOT_PATH):
    os.mkdir(ROOT_PATH)

data_paths= {}
data_paths['train'] = TRAIN_DATA_PATH
data_paths['val']   = VAL_DATA_PATH
data_paths['test']  = TEST_DATA_PATH
data_paths['data']  = DATA_PATH


for dataset_str in data_paths:
    
    data_path = data_paths[dataset_str]
    
    #make folder, if it does not exist
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        
    #clear the content of the folder
    files = glob.glob(data_path + '*')
    for f in files:
        os.remove(f)


number_of_all_available_images = 0 # this should be about 1 million
# counts the number of all available images
for pt_file_id in range(MAX_NUMBER_OF_PT_FILES):
    tensor_img_dsc_path = data_paths['data']
    tensor_img_src_path = ELOI_DATA_PATH
    tensor_img_src_name = f"{pt_file_id}.pt"
    loaded_tensor = torch.load(tensor_img_src_path + tensor_img_src_name)
    tensor_imgs = loaded_tensor['observations']
    batch_size = tensor_imgs.size(0)
    number_of_all_available_images += batch_size

# generation of image ids (image indices) from the number_of_all_available_images
# only pick a random subset of size MAX_TOTAL_IMAGE_NUMBER
# and sort the array of those randomly picket subset of image ids
all_available_image_ids = np.arange(number_of_all_available_images)
np.random.seed(SEED)
shuffled_all_available_image_ids = all_available_image_ids.copy()
np.random.shuffle(shuffled_all_available_image_ids)
shuffled_all_available_image_ids = shuffled_all_available_image_ids[:MAX_TOTAL_IMAGE_NUMBER]
shuffled_all_available_image_ids = np.sort(shuffled_all_available_image_ids)
image_ids = shuffled_all_available_image_ids
np.save(ROOT_PATH+"all_nonshuffled_image_ids.npy", image_ids)


# subsampling of MAX_TOTAL_IMAGE_NUMBER image number of images from the original dataset
image_id = 0
counter_ = 0
for pt_file_id in range(MAX_NUMBER_OF_PT_FILES):
    # dsc
    tensor_img_dsc_path = data_paths['data']
    # src
    tensor_img_src_path = ELOI_DATA_PATH
    tensor_img_src_name = f"{pt_file_id}.pt"
    # load batch of images as loaded_tensor
    loaded_tensor = torch.load(tensor_img_src_path + tensor_img_src_name)
    tensor_imgs = loaded_tensor['observations']
    # iterate over the loaded batch
    for batch_idx in range(tensor_imgs.size(0)):
        # if the current image id is in the image ids take that picture and save it at the desired location
        if image_id in image_ids:
            tensor_img_dsc_name = 'color_img_' + str(image_id).zfill(len(str(MAX_TOTAL_IMAGE_NUMBER))) + ".png"
            current_img = tensor_imgs[batch_idx, :, :, :].view(C,H,W).cpu().permute(1, 2, 0).numpy() #HWC
            plt.imsave(tensor_img_dsc_path+tensor_img_dsc_name, current_img)
            counter_ += 1
        
        #torch.save(tensor_imgs[batch_idx, :, :, :], tensor_img_dsc_path+tensor_img_dsc_name)
        #image_ids.append(image_id)
        # increase the image id coutner
        image_id += 1
        
        # print out the progress
        if counter_ % 10000 == 0:
            with open(LOGGER_PATH, 'a') as f:
                f.write(f"{counter_}/{MAX_TOTAL_IMAGE_NUMBER} image generated! ({counter_ / MAX_TOTAL_IMAGE_NUMBER * 100 : .1f}%).\n")

# data train/val/test splitting
np.random.seed(SEED)
shuffled_image_ids = image_ids.copy()
np.random.shuffle(shuffled_image_ids)
N = len(shuffled_image_ids)
# secure no overlap with train, validation and test datasets
#75% of the dataset
train_shuffled_image_ids = shuffled_image_ids[:int(N* train_dataset_percentage/100)]
#12.5% of the dataset
val_shuffled_image_ids = shuffled_image_ids[int(N* train_dataset_percentage/100):int(N* (train_dataset_percentage/100 + val_dataset_percentage/100))]
#12.5% of the dataset
test_shuffled_image_ids = shuffled_image_ids[int(N* (train_dataset_percentage/100 + val_dataset_percentage/100)):]
#check that everything adds up
assert(N == len(train_shuffled_image_ids) + len(val_shuffled_image_ids) + len(test_shuffled_image_ids))
# save the split
np.save(ROOT_PATH+"train_shuffled_image_ids.npy", train_shuffled_image_ids)
np.save(ROOT_PATH+"val_shuffled_image_ids.npy", val_shuffled_image_ids)
np.save(ROOT_PATH+"test_shuffled_image_ids.npy", test_shuffled_image_ids)


#cut operation from DATA to either DATA_TEST or DATA_VALIDATE or DATA_TEST
shuffled_image_ids = {'train':train_shuffled_image_ids, 
                      'val':val_shuffled_image_ids, 
                      'test':test_shuffled_image_ids}
for image_id in image_ids:
    img_src_path = data_paths['data']
    img_src_name = 'color_img_' + str(image_id).zfill(len(str(MAX_TOTAL_IMAGE_NUMBER))) + ".png"
    for dataset_str in ['train', 'val', 'test']:
        if image_id in shuffled_image_ids[dataset_str]:
            img_dsc_path = data_paths[dataset_str]
            img_dsc_name = img_src_name # keep the name same
            #cut operation from DATA to either DATA_TEST or DATA_VALIDATE or DATA_TEST
            os.rename(src=img_src_path+img_src_name,
                      dst=img_dsc_path+img_dsc_name)
