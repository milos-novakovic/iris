from image_generator import *

def get_image_binary_code(image_id : int, THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE:int) -> list:
    # 0 <= image_id <= (2**THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE)-1
    # if image_id = 3
    # and THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE = 4
    # output is  reverse([0,0,1,1]) = [1,1,0,0]
    image_id_binary_str = format(image_id, f'0{THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE}b')
    image_id_binary_list_int = [int(x) for x in image_id_binary_str]
    return image_id_binary_list_int[::-1]


def get_shape_specific_stats(image_binary_code : list, shape_generic_stats : dict, COLOR_DICT_WORD_2_BGR_CODE):
    # dict shape_generic_stats that has
    # key as the string named (shape_stat_name)
    # value is the array of all the possible values that particual shape_stat_name can take
    # e.g.  key shape_stat_name is equal to 'shape_name'
    #       and the values are ["Ellipse", "Parallelogram"]
    
    shape_specific_stats = {}
    bit_counter = 0
    
    #for shape_stat_name in shape_generic_stats:
    # THIS IS THE CODING ORDER
    # LSB 1 bit = 'shape_thickness'
    # next SecondLSB 1 bit = 'shape_name'
    # 2 bits = 'shape_center_x'
    # 2 bits = 'shape_center_y'
    # 2 bits = 'shape_color'
    # 2 bits = 'a'
    # 2 bits = 'b'
    # 2 bits = 'alpha'
        
    for shape_stat_name in ['shape_thickness', 'shape_name', 'shape_center_x', 'shape_center_y', 'shape_color', 'a', 'b', 'alpha']:
        if 'shape_id' == shape_stat_name:
            continue
        
        shape_stat_bit_number = np.int64(np.log2(len(shape_generic_stats[shape_stat_name])))
        
        # ['0', '1', '1']
        shape_stat_binary_code = image_binary_code[bit_counter: bit_counter + shape_stat_bit_number]
        # '011'
        shape_stat_binary_code = ''.join([str(bit_) for bit_ in shape_stat_binary_code])
        # 6
        shape_stat_binary_code = int(shape_stat_binary_code[::-1],2)
        
        # select the specific shape according to the shape_stat_binary_code
        shape_specific_stats[shape_stat_name] = shape_generic_stats[shape_stat_name][shape_stat_binary_code]
        
        if shape_stat_name == 'shape_color':
            shape_specific_stats[shape_stat_name] = COLOR_DICT_WORD_2_BGR_CODE[shape_specific_stats[shape_stat_name]]        
        
        # increment the coutner by the number of bits requested for the shape_stat_name
        bit_counter += shape_stat_bit_number
    
    # # one example of encoding a determionistic image generation
    # if [
    #     0,      #code for -> 'shape_thickness':  -1
    #     1,      #code for -> 'shape_name': 'Parallelogram'
    #     1, 1,   #code for -> 'shape_center_x': 54 # = 0.75 * W + coor_begin_x
    #     1, 0,   #code for -> 'shape_center_y': 22 # = 0.5 * H + coor_begin_y
    #     0, 0,   #code for -> 'shape_color': (255, 0, 0)
    #     0, 1,   #code for -> 'a': 12 # = 0.125 * W
    #     1, 1,   #code for -> 'b': 16 # = 0.25 * H
    #     0, 1    #code for -> 'alpha':60
    #     ] == image_binary_code:
    #     ## image_id = 11806 = = '0b10111000011110' = reverse('01111000011101') = reverse([0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1]) = reverse(image_binary_code)
    #     print(image_binary_code)
    #     print(shape_specific_stats)
    return shape_specific_stats
    
    
def train_val_test_split(data_folder_path : str, # = '/home/novakovm/DATA'
                        train_folder_path : str, # = '/home/novakovm/DATA_TRAIN'
                        val_folder_path : str,   # = '/home/novakovm/DATA_VALIDATE'
                        test_folder_path : str,  # = '/home/novakovm/DATA_TEST'
                        train_val_test_split_indices:dict, # {'train':(6/8)*2**L,'val':(7/8)*2**L,'test':(8/8)*2**L}
                        image_ids : np.array,
                        SEED : int):
    np.random.seed(SEED)
    np.random.shuffle(image_ids)
    shuffled_image_ids = image_ids
    N = len(shuffled_image_ids)
    
    # secure no overlap with train, validation and test datasets
    
    train_shuffled_image_ids = shuffled_image_ids[  0
                                                    :train_val_test_split_indices['train']]
    
    val_shuffled_image_ids = shuffled_image_ids[    train_val_test_split_indices['train']
                                                    :train_val_test_split_indices['val']]
    
    test_shuffled_image_ids = shuffled_image_ids[   train_val_test_split_indices['val']
                                                    :train_val_test_split_indices['test']]
    
    #path = '/home/novakovm/DATA_TEST' #current_working_absoulte_path + '/DATA_TEST'
    
    # clear train, validation and test folders
    os.system('rm -rf %s/*' % train_folder_path)
    os.system('rm -rf %s/*' % val_folder_path)
    os.system('rm -rf %s/*' % test_folder_path)
        
    # cut training data from from DATA to DATA_TRAIN
    img_dsc_path = train_folder_path
    for train_image_id in train_shuffled_image_ids:
        img_name = 'color_img_' + str(train_image_id).zfill(len(str(N))) + '.png'
        img_src_path =  data_folder_path + '/' + img_name        
        os.system('cp ' + img_src_path + ' ' + img_dsc_path)
        os.system('rm ' + img_src_path)
        
    # cut validation data from from DATA to DATA_VALIDATE
    img_dsc_path = val_folder_path
    for val_image_id in val_shuffled_image_ids:
        img_name = 'color_img_' + str(val_image_id).zfill(len(str(N))) + '.png'
        img_src_path =  data_folder_path + '/' + img_name        
        os.system('cp ' + img_src_path + ' ' + img_dsc_path)
        os.system('rm ' + img_src_path)
        
    # cut test data from from DATA to DATA_TEST
    img_dsc_path = test_folder_path
    for test_image_id in test_shuffled_image_ids:
        img_name = 'color_img_' + str(test_image_id).zfill(len(str(N))) + '.png'
        img_src_path =  data_folder_path + '/' + img_name        
        os.system('cp ' + img_src_path + ' ' + img_dsc_path)
        os.system('rm ' + img_src_path)
        
    # Only files left in folder '/home/novakovm/DATA' are:
    # all_generated_shapes.csv
    # stats.png
    return train_shuffled_image_ids, val_shuffled_image_ids, test_shuffled_image_ids
################

START_TIME = time.time()
#if __name__ == "__main__":

file_info_dict : dict = {key:val for one_info in data['file_info'] for key,val in one_info.items()}
H,W = file_info_dict['H'], file_info_dict['W']


#TOTAL_NUMBER_OF_IMAGES = TRAIN_TOTAL_NUMBER_OF_IMAGES# if TRAIN_MODE else TEST_TOTAL_NUMBER_OF_IMAGES

for image_id in range(TOTAL_NUMBER_OF_IMAGES):
    if image_id % 1000 == 0 and image_id > 0:
        print(f"Generated {image_id}-th image!")
    # image image_id info
    # SEED is set here
    # SEED = image_id#52#42
    # if TEST_MODE:
    #     SEED += TRAIN_TOTAL_NUMBER_OF_IMAGES     
    # np.random.seed(SEED)
    
    file_info_dict : dict = {key:val for one_info in data['file_info'] for key,val in one_info.items()}
    # if TEST_MODE:
    #     file_info_dict['file_path'] = file_info_dict['TEST_file_path']
        
    file_info_dict['image_objects'] = {'background_color' : np.array([BACKGROUND_COLOR], dtype=UINT)}    
    file_info_dict['file_version'] = '_' + str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES)))
    
    generated_image = GeneratedImage(file_info_dict)
    generated_image.generate_image()

    #images_info_dict : dict = {key:val for one_info in data['images_info'] for key,val in one_info.items()}
    #images_info_dict['shape_ids'] = np.arange(1,TOTAL_NUMBER_OF_SHAPES)
    list_of_shapes = ShapeList()

    # coordinates of all possible centers of shapes
    Y_CENTER_SPACE_np = np.round(file_info_dict['H']*COOR_BEGIN[1] + np.array(Y_CENTER_SPACE) * file_info_dict['H'], 0).astype(int)
    X_CENTER_SPACE_np = np.round(file_info_dict['W']*COOR_BEGIN[0] + np.array(X_CENTER_SPACE) * file_info_dict['W'], 0).astype(int)

    # lengths of all possible dimensions of shapes
    b_CENTER_SPACE_np = np.round(np.array(b_CENTER_SPACE) * file_info_dict['H'], 0).astype(int)
    a_CENTER_SPACE_np = np.round(np.array(a_CENTER_SPACE) * file_info_dict['W'], 0).astype(int)

    # angle of skewness (for Parallelogram) and rotation (for Ellipse)
    alpha_CENTER_SPACE_np = np.array(alpha_CENTER_SPACE).astype(int)

    # fills or no fills
    FILL_NOFILL_np = np.array(FILL_NOFILL).astype(int)

    all_shapes_variable_data = {c : [] for c in COLUMNS}
    
    ## Shape Generic stats
        
    shape_generic_stats  = {}
    #shape_generic_stats['shape_id'] = 0
    shape_generic_stats['shape_center_x'] = X_CENTER_SPACE_np
    shape_generic_stats['shape_center_y'] = Y_CENTER_SPACE_np
    shape_generic_stats['shape_color'] = COLOR_LIST# color is in words (e.g. 'red')
    #kwargs_shape['shape_color'] = COLOR_DICT_WORD_2_BGR_CODE[shape_color] 
    shape_generic_stats['shape_thickness'] = FILL_NOFILL_np # 5 = # Line thickness of 5 px
    shape_generic_stats['shape_name'] = SHAPE_TYPE_SPACE
    shape_generic_stats['shape_center_x'] = X_CENTER_SPACE_np
    shape_generic_stats['a'] = a_CENTER_SPACE_np
    shape_generic_stats['b'] = b_CENTER_SPACE_np
    shape_generic_stats['alpha'] = alpha_CENTER_SPACE_np    
    #TO DO: shape_generic_stats['shape_rotation_angle']
    #TO DO: shape_generic_stats['shape_scale_size']

    image_binary_code = get_image_binary_code(image_id=image_id,
                                              THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE=THEORETICAL_MAX_NUMBER_OF_BITS_TO_ENCODER_AN_IMAGE)

    shape_specific_stats = get_shape_specific_stats(image_binary_code=image_binary_code,
                                                    shape_generic_stats=shape_generic_stats,
                                                    COLOR_DICT_WORD_2_BGR_CODE=COLOR_DICT_WORD_2_BGR_CODE)
    # TODO
    shape_specific_stats['shape_rotation_angle'] = None
    shape_specific_stats['shape_scale_size']= None
    
    # TODO
    shape_specific_stats['image_binary_code'] = ''.join([str(x) for x in image_binary_code])
    
    for i in range(TOTAL_NUMBER_OF_SHAPES):
        #simple id (i.e. the order of shape creation)
        shape_specific_stats['shape_id'] = i
        
        kwargs_shape = {}
        # # image info (to be implemented)
        # #kwargs_shape['image_handle'] = None
        
        # # shape info
        # #kwargs_shape['shape_handle'] = None
        
        # #simple id (i.e. the order of shape creation)
        # kwargs_shape['shape_id'] = i
        
        # # random parameters to be changed/generated
        # kwargs_shape['shape_center_x'] = np.random.choice(X_CENTER_SPACE_np, size = 1)[0]
        # kwargs_shape['shape_center_y'] = np.random.choice(Y_CENTER_SPACE_np, size = 1)[0]
        
        # kwargs_shape['shape_rotation_angle'] = None
        # kwargs_shape['shape_scale_size'] = None
        
        # shape_color : str = np.random.choice( COLOR_LIST, size = 1)[0] # color is in words (e.g. 'red')
        # kwargs_shape['shape_color'] = COLOR_DICT_WORD_2_BGR_CODE[shape_color]
        # kwargs_shape['shape_thickness'] = np.random.choice(FILL_NOFILL_np, size = 1)[0] # 5 = # Line thickness of 5 px
        # kwargs_shape['shape_name'] = np.random.choice(SHAPE_TYPE_SPACE, size=1)[0] # print(np.random.choice(prog_langs, size=10, replace=True, p=[0.3, 0.5, 0.0, 0.2]))
        
        
        # if kwargs_shape['shape_name'] == "Ellipse":
        #     kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
        #     kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
        #     kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] # default 0 to form a non-rotated Ellipse
        # elif kwargs_shape['shape_name'] == "Parallelogram":
        #     kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
        #     kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
        #     kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] # default 90 to form a rectangle
        # else:
        #     assert(False, 'The shape must be Ellipse or Parallelogram!')
        
        ### SWAP kwargs_shape with deterministicaly generated shape stats
        kwargs_shape = shape_specific_stats 
        ### SWAP kwargs_shape with deterministicaly generated shape stats
        
        
        # fill in pandas df
        for c in COLUMNS:
            all_shapes_variable_data[c].append(kwargs_shape[c])
        
        new_shape = list_of_shapes.create_and_add_shape(kwargs_shape)
        generated_image.add_shape(shape=new_shape)
        #generated_image.add_shape_from_list(index_=i, list_of_shapes=list_of_shapes)

    generated_image.draw_grid_on_image( X_coors=X_CENTER_SPACE_np, 
                                        Y_coors=Y_CENTER_SPACE_np,
                                        X_CENTER_SPACE_np = X_CENTER_SPACE_np, 
                                        Y_CENTER_SPACE_np = Y_CENTER_SPACE_np,
                                        draw_lines = DRAW_IMAGINARY_LINES,
                                        draw_circles = DRAW_IMAGINARY_CIRCLES)
    

    if GENERATE_STATS:
        # create pandas df
        DF_all_shapes_variable_data = pd.DataFrame.from_dict(all_shapes_variable_data)
        
        # append newly sampled image to the olderones
        DF_all_shapes_variable_data['shape_id'] = DF_all_shapes_variable_data.index.values
        DF_all_shapes_variable_data['image_id'] = file_info_dict['file_version'][1:]
        DF_all_shapes_variable_data['shape_color_word'] = DF_all_shapes_variable_data.apply(lambda row:  COLOR_DICT_BGR_2_WORD_CODE['-'.join([str(x) for x in row['shape_color']])] , axis = 1)
        DF_all_shapes_variable_data= DF_all_shapes_variable_data[['image_id', 'shape_id'] + COLUMNS + ['shape_color_word']]
        DF_all_shapes_variable_data.to_csv(CSV_FILE_PATH, mode='a', index=False, header= not(os.path.isfile(CSV_FILE_PATH)) )


### TRAIN, VALIDATION, TEST Split
image_ids = np.arange(0, TOTAL_NUMBER_OF_IMAGES)
SEED = 0
train_shuffled_image_ids, val_shuffled_image_ids, test_shuffled_image_ids = \
train_val_test_split(   data_folder_path = data_folder_path,
                        train_folder_path = train_folder_path,
                        val_folder_path = val_folder_path,
                        test_folder_path = test_folder_path,
                        train_val_test_split_indices = train_val_test_split_indices,
                        image_ids = image_ids,
                        SEED = SEED)

np.save(main_folder_path+'/train_shuffled_image_ids.npy', train_shuffled_image_ids)
np.save(main_folder_path+'/val_shuffled_image_ids.npy', val_shuffled_image_ids)
np.save(main_folder_path+'/test_shuffled_image_ids.npy', test_shuffled_image_ids)

df_train_image_ids = pd.DataFrame(data={'image_id':train_shuffled_image_ids})
df_val_image_ids = pd.DataFrame(data={'image_id':val_shuffled_image_ids})
df_test_image_ids = pd.DataFrame(data={'image_id':test_shuffled_image_ids})

###

if GENERATE_STATS:
    for specific_df, specific_name in zip([None, df_train_image_ids, df_val_image_ids, df_test_image_ids], ['', '_train','_val','_test']):
        # read the CSV file where there is data for every generated file
        DF_all_shapes_variable_data = pd.read_csv(CSV_FILE_PATH)
        
        # init stats.png figure
        fig, axs = plt.subplots(3, 3, figsize=(20, 10))
        
        # get stats file name and file path
        STATS_FILE_NAME = f'stats{specific_name}.png'
        STATS_FILE_PATH = '/home/novakovm/DATA/' + STATS_FILE_NAME#'./DATA/stats.png'
    
        if specific_name != '':
            # filter out only specific dataset to generate stats.png file for
            DF_all_shapes_variable_data =\
                                        specific_df.merge(right = DF_all_shapes_variable_data, 
                                                    how = 'left',
                                                    on = 'image_id')\
                                                .sort_values('image_id')\
                                                .reset_index()\
                                                .drop('index',axis=1)
    
        # create histograms
        for i_col, col in enumerate(COLUMNS):
            
            col_space = None
            bar_width = None
            
            if col == 'shape_color':
                i,j = 0,1
                # colors (and names) # string values
                df_color_hist = DF_all_shapes_variable_data[['shape_color_word']].apply(pd.value_counts).reset_index()
                #df_color_hist['shape_color_word'] = df_color_hist.apply(lambda row:  COLOR_DICT_BGR_2_WORD_CODE['-'.join([str(x) for x in row['index']])] , axis = 1)#.to_frame(name = 'shape_color_word')
                #df_color_hist.drop(['index'], axis=1, inplace=True)
                df_color_hist.set_index('shape_color_word', inplace=True)
                df_color_hist = df_color_hist.T.reset_index().drop(['index'], axis=1)
                
                colors_count = df_color_hist.columns
                colors_names = df_color_hist.values[0]
                colors_number = df_color_hist.shape[1]
                y_values  = np.array(list(colors_count))
                

                #plt.figure(i_col)
                axs[i, j].set_title('Histogram of Shape Colors')
                axs[i, j].bar(np.arange(colors_number), colors_count, align='center', width=0.5, color=colors_names, edgecolor='black')
                axs[i, j].set_xticks(np.arange(colors_number), colors_names, size='small')
                #axs[i, j].set_yticks(np.arange(1, 1 + np.max(colors_count), HIST_Y_TICKS_STEP_SIZE), np.arange(1, 1 + np.max(colors_count), HIST_Y_TICKS_STEP_SIZE), size='small')
                #axs[i, j].set_yticks(y_values,y_values, size='small')
                axs[i, j].grid(which='both')
                
                for x_value, y_value in zip(np.arange(colors_number), y_values):
                    axs[i, j].text(x=x_value, y=y_value, s = y_value, ha="center", va="bottom")
                
                #plt.savefig(f'DATA/{col}_stats.png')
                continue
            elif col == 'shape_name':
                col_space = SHAPE_TYPE_SPACE
                bar_width = len(col_space)*5*1e-2
                i,j = 0,0
                xlabel, ylabel, title = f'Possible names', '#', f'Histogram of names'
                color = 'black'
            elif col == 'a':
                col_space = a_CENTER_SPACE_np
                bar_width = np.max(col_space)*5*1e-2
                i,j = 1,0
                color = 'blue'
                xlabel, ylabel, title = f'Possible *a*', '#', f'Histogram of *a*'
            elif col == 'b':
                col_space = b_CENTER_SPACE_np
                bar_width = np.max(col_space)*5*1e-2
                xlabel, ylabel, title = f'Possible *b*', '#', f'Histogram of *b*'
                i,j = 1,1
                color = 'blue'
            elif col == 'alpha':
                col_space = alpha_CENTER_SPACE_np
                bar_width = np.max(col_space)*5*1e-2
                i,j = 1,2
                xlabel, ylabel, title = f'Possible *alpha*', '#', f'Histogram of *alpha*'
                color = 'blue'
            elif col == 'shape_center_x':
                col_space = X_CENTER_SPACE_np
                bar_width = np.max(col_space)*5*1e-2
                i,j = 2,0
                xlabel, ylabel, title = f'Possible *x_center*', '#', f'Histogram of *x_center*'
                color = 'orange'
            elif col == 'shape_center_y':
                col_space = Y_CENTER_SPACE_np
                bar_width = np.max(col_space)*5*1e-2
                i,j = 2,1
                xlabel, ylabel, title = f'Possible *y_center*', '#', f'Histogram of *y_center*'
                color = 'orange'
            else:
                continue
            
            # numerical values
            df_column_hist = DF_all_shapes_variable_data[[col]].apply(pd.value_counts).reset_index()
            df_column_hist.rename({'index':'count_'+col}, axis =1, inplace=True)
            df_column_hist.sort_values(by = ['count_'+col], inplace=True)
            df_column_hist.set_index('count_'+col, inplace=True)
            df_column_hist = df_column_hist.T.reset_index().drop(['index'], axis=1)

            #plt.figure(i_col)
            axs[i, j].set_title(title)
            axs[i, j].bar(df_column_hist.columns.values, df_column_hist.values[0], align='center', width=bar_width, color=color, edgecolor='black')
            axs[i, j].set_xticks(col_space, col_space, size='small')
            #axs[i, j].set_yticks(df_column_hist.values[0], df_column_hist.values[0], size='small')
            #axs[i, j].set_yticks(np.arange(1, 1 + np.max(df_column_hist.values[0]), HIST_Y_TICKS_STEP_SIZE), np.arange(1, 1 + np.max(df_column_hist.values[0]), HIST_Y_TICKS_STEP_SIZE), size='small')
            axs[i, j].grid(which='both')
            axs[i, j].set_xlabel(xlabel)
            axs[i, j].set_ylabel(ylabel)
            
            for x_value, y_value in zip(col_space, df_column_hist.values[0]):
                axs[i, j].text(x=x_value, y=y_value, s = y_value, ha="center", va="bottom")
                #axs[i, j].text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom")
            # for bar in bars:
            #     yval = bar.get_height()
            #     plt.text(bar.get_x(), yval + .005, yval)
            #plt.savefig(f'DATA/{col}_stats.png')

        # delete unused subplots
        fig.delaxes(axs[0,2])
        fig.delaxes(axs[2,2])
        plt.tight_layout()
        plt.savefig(STATS_FILE_PATH)
        # copy stats.png image to main folder
        os.system('cp ' + STATS_FILE_PATH + ' ' + main_folder_path)
    # copy all_generated_shapes.csv image to main folder
    os.system('cp ' + CSV_FILE_PATH + ' ' + main_folder_path)
    


    
# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()



# calculate the number of bits required for exactly one shape
number_of_bits_required_for_one_shape = sum([np.log2(1.0 * len(space))\
                                            for space in \
                                            [SHAPE_TYPE_SPACE,\
                                            COLOR_LIST,\
                                            Y_CENTER_SPACE,\
                                            X_CENTER_SPACE,\
                                            b_CENTER_SPACE,\
                                            a_CENTER_SPACE,\
                                            alpha_CENTER_SPACE,\
                                            FILL_NOFILL]])

# calculate the number of bits required for all shapes combined
number_of_bits_required_for_one_image = TOTAL_NUMBER_OF_SHAPES * number_of_bits_required_for_one_shape

print(f"Number of bits for one SHAPE = {number_of_bits_required_for_one_shape}b.")
print(f"Number of SHAPEs = {TOTAL_NUMBER_OF_SHAPES}.")
print(f"Number of bits for one IMAGE = {number_of_bits_required_for_one_image}b.")
print(f"Total number of seconds that the program runs = {round(time.time() - START_TIME,2)} sec.")

debug = 0