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

CSV_FILE_PATH = './DATA/all_generated_shapes.csv'
STATS_FILE_PATH = './DATA/stats.png'
GENERATE_STATS = False

current_working_absoulte_path = '/home/novakovm/iris/MILOS'
os.chdir(current_working_absoulte_path)

milos_config_path = 'milos_config.yaml'
# Open the file and load the file
with open(milos_config_path) as f:
    data = yaml.load(f, Loader=SafeLoader)

COOR_BEGIN =                   [one_info for one_info in data['COOR_BEGIN']]
SHAPE_TYPE_SPACE =             [one_info for one_info in data['SHAPE_TYPE_SPACE']]
COLOR_LIST =                   [one_info for one_info in data['COLOR_LIST']]
Y_CENTER_SPACE =               [one_info for one_info in data['Y_CENTER_SPACE']]
X_CENTER_SPACE =               [one_info for one_info in data['X_CENTER_SPACE']]
b_CENTER_SPACE =               [one_info for one_info in data['b_CENTER_SPACE']]
a_CENTER_SPACE =               [one_info for one_info in data['a_CENTER_SPACE']]
alpha_CENTER_SPACE =           [one_info for one_info in data['alpha_CENTER_SPACE']]
FILL_NOFILL =                  [one_info for one_info in data['FILL_NOFILL']]
TOTAL_NUMBER_OF_SHAPES =       [dict_['TOTAL_NUMBER_OF_SHAPES'] for dict_ in data['file_info'] if 'TOTAL_NUMBER_OF_SHAPES' in dict_][0]
TOTAL_NUMBER_OF_IMAGES =       [dict_['TOTAL_NUMBER_OF_IMAGES'] for dict_ in data['file_info'] if 'TOTAL_NUMBER_OF_IMAGES' in dict_][0]


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