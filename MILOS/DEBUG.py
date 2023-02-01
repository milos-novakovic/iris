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

# because of TensorFlow warning, see here: https://stackoverflow.com/questions/60368298/could-not-load-dynamic-library-libnvinfer-so-6
os.system("export TF_CPP_MIN_LOG_LEVEL=2")

config_folder_names = { "toy_dataset" : "/home/novakovm/iris/MILOS/",
                        "crafter_dataset" : "/home/novakovm/iris/MILOS/"}

config_file_names = {   "toy_dataset" : "toy_shapes_config",
                        "crafter_dataset" : "crafter_config"}

config_full_file_paths = {}
for dataset_name in config_file_names:
    config_full_file_paths[dataset_name] = config_folder_names[dataset_name] + \
                                           config_file_names[dataset_name] + ".yaml"


K_BIT_MIN, K_BIT_MAX = 8,10 #2,3 #4,7 #
D_array = np.array([32]) #np.array([32, 16, 8]) #
M_array = np.array([7, 3]) #np.array([3]) #M_array = -np.sort(-M_array)
K_array = 2** np.arange(K_BIT_MIN, K_BIT_MAX+1)


K_BIT_MIN, K_BIT_MAX = 11,15 #2,3 #4,7 #
D_array = np.array([32]) #np.array([32, 16, 8]) #
M_array = np.array([7, 3]) #np.array([3]) #M_array = -np.sort(-M_array)
K_array = 2** np.arange(K_BIT_MIN, K_BIT_MAX+1)


##
## KEEPING VQ_BITS CONST 
##
## bits (M,log2 K)1  (M,log2 K)2
# 128    8,  2        4,   8
# 192    8,  3        4,   12
# 256    8,  4        4,   16
# 320    8,  5        4,   20
# 
# ##
#D_array = np.array([32]) #np.array([32, 16, 8]) #
#M_array = np.array([3]) 
#K_array = 2 ** np.array([12,16,20])

#D_array = np.array([32]) #np.array([32, 16, 8]) #
#M_array = np.array([7]) 
#K_array = 2 ** np.array([2,3,4])

# trying different pairs of M&K for same number of bits
# M_K_D_array = [ (7, 2**1,  16),  # bits = 64
#                 (3, 2**4,  16),  # bits = 64
                
#                 (7, 2**2,  16),  # bits = 128 # does not learn!
#                 (3, 2**8,  16),  # bits = 128
                
#                 (7, 2**3,  16),  # bits = 192
#                 (3, 2**12, 16),  # bits = 192
                
#                 (7, 2**4,  16),  # bits = 256
#                 (3, 2**16, 16)  # bits = 256
                
#                 #(7, 2**5,  8),  # bits = 320#too large
#                 #(3, 2**20, 8)   # bits = 320#too large
#                 ]

M_K_D_array = [(3, 2**15, 32)]
 
#M_K_D_array = np.array(M_K_D_array)[::-1]

# M_K_D_array = [ #(7, 2**9,  32),  
#                 (3, 2**9,  32)# for the case with a lot of parameters max_channel_number : 256 & divisor_value : 1
#                 ] 

#max_channel_number_array = np.array([256])
#divisor_value_array      = np.array([1])



#M_K_D_array = [(3, 2**15, 4)]# radi ali  sporo!
#M_K_D_array = [(3, 2**19, 8)]# radi ali bas bas sporo!


# Trying different Ds
# M_K_D_array = M_K_D_array + \
#             [(7, 2**7, 1),
#              (7, 2**7, 2),
#              (7, 2**7, 4)]



#K_array = -np.sort(-K_array)
#max_channel_number = np.array([256])

trainers = {}
custom_logger= "/home/novakovm/iris/MILOS/CUSTOM_LOGGER.txt"

# init values for run_ids
RUN_ID_DELTA_START = 83#53#80#43#37 #37#1
run_id = { "toy_dataset" : 100 + RUN_ID_DELTA_START,
        "crafter_dataset" : 500 + RUN_ID_DELTA_START}


USE_PRETRAINED_MODEL_run_id_to_get_current_time_str = {
                                582 : "2023_01_30_22_56_40",# 4x4; D=32	log2(K)=9	M=4	bits = 144 (30 mil. params.)
                                581 : "2023_01_28_07_36_14",# 4x4; D=32	log2(K)=9	M=4	bits = 144 (7 mil. params.)
                                539 : "2023_01_26_03_18_43",#iris 8x8 = D=32 log2(K)=9 M=8 bits = 576 
                                552 : "2023_01_27_06_48_21", # 4x4; D=32	log2(K)=15	M=4	bits = 240
                                # two-shapes
                                101 : "2023_01_22_08_56_42",
                                102 : "2023_01_22_11_28_18",
                                103 : "2023_01_22_13_51_02",
                                104 : "2023_01_22_15_55_05",
                                105 : "2023_01_22_17_52_54",
                                106 : "2023_01_22_19_49_58",
                                107 : "2023_01_22_21_28_59",
                                108 : "2023_01_22_23_00_05",
                                109 : "2023_01_23_00_31_06",
                                110 : "2023_01_23_02_37_02",
                                111 : "2023_01_23_04_39_19",
                                112 : "2023_01_23_06_41_39",
                                113 : "2023_01_23_08_17_02",
                                114 : "2023_01_23_10_15_54",
                                115 : "2023_01_23_12_18_32",
                                116 : "2023_01_23_14_11_21",
                                117 : "2023_01_23_15_48_16",
                                118 : "2023_01_23_17_19_28",
                                119 : "2023_01_23_19_27_06",
                                120 : "2023_01_23_21_31_36",
                                121 : "2023_01_23_23_33_41",
                                122 : "2023_01_24_01_04_49",
                                123 : "2023_01_24_02_42_48",
                                124 : "2023_01_24_04_40_07",
                                125 : "2023_01_24_06_37_48",
                                126 : "2023_01_24_08_32_06",
                                127 : "2023_01_24_10_06_28",
                                128 : "2023_01_24_12_13_54",
                                129 : "2023_01_24_14_20_38",
                                130 : "2023_01_24_16_24_18",
                                131 : "2023_01_24_17_55_44",
                                132 : "2023_01_24_19_26_55",
                                133 : "2023_01_24_21_07_02",
                                134 : "2023_01_24_22_59_19",
                                135 : "2023_01_25_00_53_29",
                                136 : "2023_01_25_02_51_13",
                                # crafter
                                501 : "2023_01_22_15_53_52",
                                502 : "2023_01_22_18_10_59",
                                503 : "2023_01_22_20_26_48",
                                504 : "2023_01_22_22_06_14",
                                505 : "2023_01_22_23_45_50",
                                506 : "2023_01_23_01_42_29",
                                507 : "2023_01_23_03_50_42",
                                508 : "2023_01_23_05_58_34",
                                509 : "2023_01_23_07_49_28",
                                510 : "2023_01_23_10_07_56",
                                511 : "2023_01_23_12_24_49",
                                512 : "2023_01_23_14_36_31",
                                513 : "2023_01_23_16_16_04",
                                514 : "2023_01_23_18_06_59",
                                515 : "2023_01_23_20_19_55",
                                516 : "2023_01_23_22_27_59",
                                517 : "2023_01_24_00_23_27",
                                518 : "2023_01_24_02_03_49",
                                519 : "2023_01_24_04_22_22",
                                520 : "2023_01_24_06_35_01",
                                521 : "2023_01_24_08_46_13",
                                522 : "2023_01_24_10_31_38",
                                523 : "2023_01_24_12_45_13",
                                524 : "2023_01_24_14_58_37",
                                525 : "2023_01_24_16_58_13",
                                526 : "2023_01_24_18_39_06",
                                527 : "2023_01_24_20_19_26",
                                528 : "2023_01_24_22_34_17",
                                529 : "2023_01_25_00_46_04",
                                530 : "2023_01_25_02_55_57",
                                531 : "2023_01_25_04_03_29",
                                532 : "2023_01_25_05_11_26",
                                533 : "2023_01_25_06_19_05",
                                534 : "2023_01_25_07_26_02",
                                535 : "2023_01_25_08_33_33",
                                536 : "2023_01_25_09_40_59"
                                }

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

# for K in K_array:
#     for M in M_array:
#         for D in D_array:
for M,K,D in M_K_D_array:
    # logger settings
    compressed_number_of_bits_per_image = int(np.ceil((M+1)**2 * np.log2(K)))
    current_time_str = time.strftime("%H:%M:%S %d.%m.%Y", time.localtime(time.time()))
    log_str = f"[{current_time_str}] {run_id[dataset_name]}) Running for K = {K} & D = {D} & M = {M} (i.e. bits = {compressed_number_of_bits_per_image})"
    
    # config setup
    #for config_file_name in config_file_names:
    update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                get_new_data_from_human_readable_yaml_file = True,
                key = "K", 
                value = K)
    update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                key = "M", 
                value = M)
    update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                key = "D",
                value = D)
    update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                key = "run_id", 
                value = run_id[dataset_name])
    
    # for pretrained models
    if run_id[dataset_name] in USE_PRETRAINED_MODEL_run_id_to_get_current_time_str:
        update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                    key = "USE_PRETRAINED_MODEL_run_id", 
                    value = run_id[dataset_name])# keep same run_id
        update_yaml(yaml_folder_name = config_folder_names[dataset_name], yaml_file_name = config_file_names[dataset_name],
                    key = "USE_PRETRAINED_MODEL_current_time_str", 
                    value = USE_PRETRAINED_MODEL_run_id_to_get_current_time_str[run_id[dataset_name]],
                    value_type = str)#get the current time of the pretrained model
    
    
    # inc run id
    run_id[dataset_name] += 1
    
    # if run_id[dataset_name]-1 != 528:#122:
    #   continue
    
                        
    
    # logger begin
    custom_logger_begin(custom_logger, log_str)
    # run
    os.system("python /home/novakovm/iris/MILOS/run_pipeline.py " + run_args)
    #run(config_full_file_paths[dataset_name])
    # logger end
    custom_logger_end(custom_logger, log_str)