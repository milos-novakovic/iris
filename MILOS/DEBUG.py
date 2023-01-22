import sys
from run_pipeline import run
import numpy as np
import time
from helper_functions import update_yaml
import os
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
    config_full_file_paths[dataset_name] = config_folder_names[dataset_name] + \
                                           config_file_names[dataset_name] + ".yaml"


K_BIT_MIN, K_BIT_MAX = 4,7
D_array =np.array([32, 16, 8])
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

DATASET_ARG = sys.argv[1]

if DATASET_ARG != "toy" and DATASET_ARG != "crafter":
    assert(False)


if DATASET_ARG == "toy":
    dataset_name,  dataset_type = "toy_dataset", "test"
    run_args = "/home/novakovm/iris/MILOS/toy_shapes_config.yaml"
    
elif DATASET_ARG == "crafter":
    dataset_name, dataset_type = "crafter_dataset", "test"
    run_args =  "/home/novakovm/iris/MILOS/crafter_config.yaml"

for K in K_array:
    for M in M_array:
        for D in D_array:
            # logger settings
            compressed_number_of_bits_per_image = int(np.ceil((M+1)**2 * np.log2(K)))
            current_time_str = time.strftime("%H:%M:%S %d.%m.%Y", time.localtime(time.time()))
            log_str = f"[{current_time_str}] {run_id}) Running for K = {K} & D = {D} & M = {M} (i.e. bits = {compressed_number_of_bits_per_image})"
            
            # config setup
            #for config_file_name in config_file_names:
            update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                        get_new_data_from_human_readable_yaml_file = True,
                        key = "K", value = K)
            update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                        key = "M", value = M)
            update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                        key = "D", value = D)
            update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                        key = "run_id", value = run_id[dataset_name])
            # inc run id
            run_id[dataset_name] += 1
                                
            # logger begin
            custom_logger_begin(custom_logger, log_str)
            # run
            os.system("python /home/novakovm/iris/MILOS/run_pipeline.py " + run_args)
            #run(config_full_file_paths[dataset_name])
            # logger end
            custom_logger_end(custom_logger, log_str)