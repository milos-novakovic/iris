from IPython.display import clear_output
from run_pipeline import run
import torch
import numpy as np
import imageio
import yaml
import time
def change_one_token(trainers, dataset_name = "toy_dataset", dataset_type = "test"):
    M = trainers[dataset_name].model.VQ.M
    K = trainers[dataset_name].model.VQ.K

    image_batch, image_id_batch = next(iter(trainers[dataset_name].loaders[dataset_type]))
    image_batch,image_id_batch = image_batch.to(trainers[dataset_name].device), image_id_batch.to(trainers[dataset_name].device)

    trainers[dataset_name].model.VQ.output_whole_quantization_process = True
    e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words, encoding_indices, estimate_codebook_words_freq, estimate_codebook_words_prob, inputs, D  = \
        trainers[dataset_name].model.VQ(trainers[dataset_name].model.encoder(image_batch))
    trainers[dataset_name].model.VQ.output_whole_quantization_process = False
    
    changed_token_map_position_range_step = 1
    changed_token_map_position_range = np.arange(0,K,changed_token_map_position_range_step)
    for changed_token_map_position_row in np.arange(M+1):
        for changed_token_map_position_column in np.arange(M+1):
            
            digit_size = len(str(len(trainers[dataset_name].loaders[dataset_type].dataset)))
            
            for index_, changed_token_map_position_value in enumerate(changed_token_map_position_range):
                new_encoding_indices = encoding_indices.clone().detach().view(M+1,M+1)
                new_encoding_indices[changed_token_map_position_row, changed_token_map_position_column] = changed_token_map_position_value
                new_encoding_indices = new_encoding_indices.view(-1,1)
                trainers[dataset_name].original_reconstructed_changed_reconstucted_tokens_changed_tokens(  image_batch,# one image, it is the tensor of shape = (1,C,H,W)
                                                                                                            jupyter_show_images = False)
            frames_per_second = 5#10#5#10#1
            format_list = ['mp4', 'gif']
            for format_ in format_list:
                with imageio.get_writer(trainers[dataset_name].visualize_tokens_path + f"_{str(changed_token_map_position_row).zfill(1)}_x_{str(changed_token_map_position_column).zfill(1)}_token_changed_{format_}.{format_}", mode='I', fps = frames_per_second) as writer:
                    for index_, changed_token_map_position_value in enumerate(changed_token_map_position_range):
                        filename = trainers[dataset_name].all_images_full_path + f"{str(index_).zfill(digit_size)}_custom_image.png"
                        image = imageio.imread(filename)
                        writer.append_data(image)
                    writer.close()
                    
def update_yaml(yaml_folder_name = "/home/novakovm/iris/MILOS/",
                yaml_file_name = "toy_shapes_config",
                key = "NUM_WORKERS", 
                value = 5,
                source = "training_hyperparams"):
    # make a new yaml config file from human readable version of that same file
    yaml_full_src_path = yaml_folder_name + yaml_file_name+ "_human_readable" +".yaml"
    yaml_full_dsc_path = yaml_folder_name + yaml_file_name +".yaml"
    data_dict = yaml.load(open(yaml_full_src_path, 'r'), Loader=yaml.FullLoader)
    with open(yaml_full_dsc_path, 'w') as yaml_file:
        yaml_file.write( yaml.dump(data_dict, default_flow_style=False))
    
    # update the newly created non-human readable file
    data_dict = yaml.load(open(yaml_full_dsc_path, 'r'), Loader=yaml.FullLoader)
    # pick the desired source list in the data dict. yaml file
    data_source_list = data_dict[source]
    # find the index in the list where the desired key is
    data_source_list_key_index = [idx for idx, data_key in enumerate(data_source_list) if key in data_key][0]
    # update step
    data_source_list[data_source_list_key_index][key] = value
    # save the change
    with open(yaml_full_dsc_path, 'w') as yaml_file:
        yaml_file.write( yaml.dump(data_dict, default_flow_style=False))

def custom_logger_begin(custom_logger, log_str):
    with open(custom_logger, 'a') as f:
        f.write(f"****************************************************************************************************************\n")
        f.write(f"----- BEGIN RUN FOR TOY SHAPES AND CRAFTER -----\n")
        f.write(f"{log_str}:\n")

def custom_logger_end(custom_logger, log_str):
    with open(custom_logger, 'a') as f:
        f.write(f"{log_str}:\n")
        f.write(f"----- END RUN FOR TOY SHAPES AND CRAFTER -----\n")
        f.write(f"****************************************************************************************************************\n\n")

config_folder_names = { "toy_dataset" : "/home/novakovm/iris/MILOS/",
                        "crafter_dataset" : "/home/novakovm/iris/MILOS/"}

config_file_names = {   "toy_dataset" : "toy_shapes_config",
                        "crafter_dataset" : "crafter_config"}

config_full_file_paths = {}
for dataset_name in config_file_names:
    config_full_file_paths[dataset_name] = config_folder_names[dataset_name] + config_file_names[dataset_name] + ".yaml"


K_BIT_MIN, K_BIT_MAX = 3,8
D_array =np.array([64, 32, 16, 8])
M_array = np.array([7, 3, 1]) #M_array = -np.sort(-M_array)
K_array = 2** np.arange(K_BIT_MIN, K_BIT_MAX+1)
K_array = -np.sort(-K_array)

trainers = {}
custom_logger= "/home/novakovm/iris/MILOS/CUSTOM_LOGGER.txt"

# init values for run_ids
run_id = { "toy_dataset" : 101,
        "crafter_dataset" : 501}

"""
Delimiter in log_all.txt-like files for this simulation
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
"""

for K in K_array:
    for M in M_array:
        for D in D_array:
            # logger settings
            compressed_number_of_bits_per_image = int(np.ceil((M+1)**2 * np.log2(K)))
            current_time_str = time.strftime("%H:%M:%S %d.%m.%Y", time.localtime(time.time()))
            log_str = f"[{current_time_str}] {run_id}) Running for K = {K} & D = {D} & M = {M} (i.e. bits = {compressed_number_of_bits_per_image})"
            
            # config setup
            for dataset_name in config_file_names:
                update_yaml(yaml_file_path = config_folder_names[dataset_name],
                            yaml_file_name = config_file_names[dataset_name],
                            key = "K", 
                            value = K)
                update_yaml(yaml_file_path = config_folder_names[dataset_name],
                            yaml_file_name = config_file_names[dataset_name],
                            key = "M", 
                            value = M)
                update_yaml(yaml_file_path = config_folder_names[dataset_name],
                            yaml_file_name = config_file_names[dataset_name],
                            key = "D", 
                            value = D)
                update_yaml(yaml_file_path = config_folder_names[dataset_name],
                            yaml_file_name = config_file_names[dataset_name],
                            key = "run_id", 
                            value = run_id[dataset_name])
                # inc run id
                run_id[dataset_name] += 1
                
                
            trainers = {}
            
            # toy dataset
            custom_logger_begin(custom_logger, log_str)
            trainers['toy_dataset'] = run(config_full_file_paths["toy_dataset"])
            change_one_token(trainers, dataset_name = "toy_dataset", dataset_type = "test")
            
            
            # crafter
            trainers['crafter_dataset'] = run(config_full_file_paths["crafter_dataset"])
            change_one_token(trainers, dataset_name = "crafter_dataset", dataset_type = "test")
            custom_logger_end(custom_logger, log_str)