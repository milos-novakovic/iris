import numpy as np
import torch.utils.data as data
from helper_functions import get_hyperparam_from_config_file
from find_mean_std import find_mean_std

config_path = "/home/novakovm/iris/MILOS/toy_shapes_config.yaml"
TRAIN_DATA_PATH =        get_hyperparam_from_config_file(config_path, 'TRAIN_DATA_PATH')
ROOT_PATH =              get_hyperparam_from_config_file(config_path, 'ROOT_PATH')
MAX_TOTAL_IMAGE_NUMBER = get_hyperparam_from_config_file(config_path, 'MAX_TOTAL_IMAGE_NUMBER')

train_shuffled_image_ids = np.load(ROOT_PATH + 'train_shuffled_image_ids.npy')

#TOTAL_NUMBER_OF_IMAGES = [dict_['TOTAL_NUMBER_OF_IMAGES'] for dict_ in data['file_info'] if 'TOTAL_NUMBER_OF_IMAGES' in dict_][0]
#train_folder_path = [dict_['train_folder_path'] for dict_ in data['file_info'] if 'train_folder_path' in dict_][0]
#main_folder_path = [dict_['main_folder_path'] for dict_ in data['file_info'] if 'main_folder_path' in dict_][0]



RGB_mean, RGB_std, RGB_mean_np, RGB_std_np, X_mean_bar_all_ch, X_std_bar_all_ch,  Total_mean_np, Total_std_np = find_mean_std(MAX_TOTAL_IMAGE_NUMBER, 
                                                                                                                                image_ids_numbers = train_shuffled_image_ids,
                                                                                                                                train_folder_path=TRAIN_DATA_PATH,
                                                                                                                                main_folder_path=ROOT_PATH)


#Step 2 - Take Sample data

#img = Image.open("./DATA/color_img_000.png")
#H,W = img.size

#Step 3 - Convert to tensor

#convert_tensor = transforms.ToTensor()

#tensor_img = convert_tensor(img)

# ch 0 is Red (R)
# ch 1 is Green (G)
# ch 2 is Blue (B)

# BGR Format for cv2
#cv2.imwrite(TRAIN_DATA_PATH + "mean_normalize_train_dataset.png", BGR_mean_normalize) 
#cv2.imwrite(TRAIN_DATA_PATH + "std_normalize_train_dataset.png",  BGR_std_normalize)

############


