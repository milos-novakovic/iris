# Python3 program to draw solid-colored
# image using numpy.zeroes() function
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
# import pyyaml module
import yaml
from yaml.loader import SafeLoader
import os
import time

INT = np.int64
FLOAT = np.float64
UINT  = np.uint8

GENERATE_STATS = True
TRAIN_MODE = True
TEST_MODE = not(TRAIN_MODE)


CSV_FILE_PATH, STATS_FILE_PATH = None, None
if TRAIN_MODE:
    CSV_FILE_PATH = '/home/novakovm/DATA/all_generated_shapes.csv'#'./DATA/all_generated_shapes.csv'
    STATS_FILE_PATH = '/home/novakovm/DATA/stats.png'#'./DATA/stats.png'
elif TEST_MODE:
    CSV_FILE_PATH = '/home/novakovm/DATA_TEST/all_generated_shapes.csv'#'./DATA_TEST/all_generated_shapes.csv'
    STATS_FILE_PATH = '/home/novakovm/DATA_TEST/stats.png'#'./DATA_TEST/stats.png'

#current_working_absoulte_path = '/home/novakovm/iris/MILOS'
#os.chdir(current_working_absoulte_path)


#rm -rf /home/novakovm/iris/MILOS/DATA/*

#import os
if TRAIN_MODE:
    # if training then delete previous dataset
    path = '/home/novakovm/DATA' #current_working_absoulte_path + '/DATA'
    #os.system('rm -rf %s/*' % path)
    os.chdir(path)
    if os.getcwd() == path:
        os.system("find . -name \"*.png\" -delete")
        os.system("find . -name \"*.csv\" -delete")
    
if TEST_MODE:
    # if training then delete previous dataset
    path = '/home/novakovm/DATA_TEST' #current_working_absoulte_path + '/DATA_TEST'
    os.system('rm -rf %s/*' % path)
    os.chdir(path)
    if os.getcwd() == path:
        os.system("find . -name \"*.png\" -delete")
        os.system("find . -name \"*.csv\" -delete")

milos_config_path = '/home/novakovm/iris/MILOS/milos_config.yaml'
# Open the file and load the file
with open(milos_config_path) as f:
    data = yaml.load(f, Loader=SafeLoader)

COOR_BEGIN =                   [one_info for one_info in data['COOR_BEGIN']]


THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES = 1
# 1 bits
SHAPE_TYPE_SPACE =             [one_info for one_info in data['SHAPE_TYPE_SPACE']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(SHAPE_TYPE_SPACE)
# 2 bits
COLOR_LIST =                   [one_info for one_info in data['COLOR_LIST']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(COLOR_LIST)
# 2 bits
Y_CENTER_SPACE =               [one_info for one_info in data['Y_CENTER_SPACE']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(Y_CENTER_SPACE)
# 2 bits
X_CENTER_SPACE =               [one_info for one_info in data['X_CENTER_SPACE']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(X_CENTER_SPACE)
# 2 bits
b_CENTER_SPACE =               [one_info for one_info in data['b_CENTER_SPACE']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(b_CENTER_SPACE)
# 2 bits
a_CENTER_SPACE =               [one_info for one_info in data['a_CENTER_SPACE']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(a_CENTER_SPACE)
# 2 bits
alpha_CENTER_SPACE =           [one_info for one_info in data['alpha_CENTER_SPACE']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(alpha_CENTER_SPACE)
# 1 bit
FILL_NOFILL =                  [one_info for one_info in data['FILL_NOFILL']]
THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES *= len(FILL_NOFILL)

#THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE = np.int64(np.ceil(np.log2(np.float64(THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES))))
THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE = np.int64(np.log2(np.float64(THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES)))
assert(2** THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE == THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES, f"{2** THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE} is not equal to {THEORETICAL_MAX_NUMBER_OF_DIFFERENT_IMAGES} ; Number of bits has to be a positive integer!")



TOTAL_NUMBER_OF_SHAPES =       [dict_['TOTAL_NUMBER_OF_SHAPES'] for dict_ in data['file_info'] if 'TOTAL_NUMBER_OF_SHAPES' in dict_][0]
TOTAL_NUMBER_OF_IMAGES =       [dict_['TOTAL_NUMBER_OF_IMAGES'] for dict_ in data['file_info'] if 'TOTAL_NUMBER_OF_IMAGES' in dict_][0]

# Test
TEST_TOTAL_NUMBER_OF_IMAGES = [dict_['TEST_TOTAL_NUMBER_OF_IMAGES'] for dict_ in data['file_info'] if 'TEST_TOTAL_NUMBER_OF_IMAGES' in dict_][0]
TRAIN_TOTAL_NUMBER_OF_IMAGES = TOTAL_NUMBER_OF_IMAGES

# [-1]['TOTAL_NUMBER_OF_SHAPES']#data['TOTAL_NUMBER_OF_SHAPES']
# TOTAL_NUMBER_OF_IMAGES =       data['file_info'][-1]['TOTAL_NUMBER_OF_IMAGES']#data['TOTAL_NUMBER_OF_SHAPES']
        
COLOR_DICT_WORD_2_BGR_CODE =   { color_word : tuple(data['COLOR_DICT_WORD_2_BGR_CODE'][color_word])
                                        for color_word in data['COLOR_DICT_WORD_2_BGR_CODE']}

# BGR code to word dict
COLOR_DICT_BGR_2_WORD_CODE = {}
for word_code in COLOR_DICT_WORD_2_BGR_CODE:
    B,G,R = COLOR_DICT_WORD_2_BGR_CODE[word_code]
    BRG_code = str(B) + '-' + str(G) + '-' + str(R) #"B-G-R" code
    COLOR_DICT_BGR_2_WORD_CODE[BRG_code] = word_code

# which shape features will be changed
# i.e. columns in pandas DF that will show the histograms of randomly distributed features
COLUMNS = [ #'shape_id',\
            'shape_name',\
            'a',\
            'b',\
            'shape_center_x',\
            'shape_center_y',\
            'alpha',\
            'shape_color',\
            #'shape_thickness'\
            ]

BACKGROUND_COLOR       =data['BACKGROUND_COLOR']
HIST_Y_TICKS_STEP_SIZE =data['HIST_Y_TICKS_STEP_SIZE']
DRAW_IMAGINARY_LINES   =data['DRAW_IMAGINARY_LINES']
DRAW_IMAGINARY_CIRCLES =data['DRAW_IMAGINARY_CIRCLES']