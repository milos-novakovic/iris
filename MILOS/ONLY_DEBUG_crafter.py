from run_pipeline import run
from helper_functions import update_yaml

#
config_folder_names = { "crafter_dataset" : "/home/novakovm/iris/MILOS/"} #"toy_dataset" : "/home/novakovm/iris/MILOS/",

#
config_file_names = {   "crafter_dataset" : "crafter_config"} # "toy_dataset" : "toy_shapes_config",

#
config_full_file_paths = {}
for dataset_name in config_file_names:
    config_full_file_paths[dataset_name] = config_folder_names[dataset_name] + \
                                           config_file_names[dataset_name] + ".yaml"
# config setup
for dataset_name in config_file_names:
    update_yaml(yaml_folder_name = config_folder_names[dataset_name],
                yaml_file_name = config_file_names[dataset_name],
                key = None, 
                value = None)
        
trainers = {}
# crafter
dataset_name, dataset_type = "crafter_dataset", "test"
trainers[dataset_name] = run(config_full_file_paths[dataset_name])
