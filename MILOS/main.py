from image_generator import *

START_TIME = time.time()




#if __name__ == "__main__":


for image_id in range(TOTAL_NUMBER_OF_IMAGES):
    file_info_dict : dict = {key:val for one_info in data['file_info'] for key,val in one_info.items()}

    # obj info

    file_info_dict['image_objects'] = {'background_color' : np.array([BACKGROUND_COLOR], dtype=UINT)}


    SEED = image_id#52#42     
    np.random.seed(SEED)
    file_info_dict['file_version'] = '_' + str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES)))
    
    
    

    first_generated_image = GeneratedImage(file_info_dict)

    first_generated_image.generate_image()

    images_info_dict : dict = {key:val for one_info in data['images_info'] for key,val in one_info.items()}

    images_info_dict['shape_ids'] = np.arange(1,TOTAL_NUMBER_OF_SHAPES)

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
    # for c in COLUMNS:
    #     all_shapes_variable_data[c] = []

    for i in range(TOTAL_NUMBER_OF_SHAPES):
        kwargs_shape = {}
        
        # image info (to be implemented)
        #kwargs_shape['image_handle'] = None
        
        # shape info
        #kwargs_shape['shape_handle'] = None
        
        #simple id (i.e. the order of shape creation)
        kwargs_shape['shape_id'] = i
        
        # random parameters to be changed/generated
        kwargs_shape['shape_center_x'] = np.random.choice(X_CENTER_SPACE_np, size = 1)[0]
        kwargs_shape['shape_center_y'] = np.random.choice(Y_CENTER_SPACE_np, size = 1)[0]
        
        kwargs_shape['shape_rotation_angle'] = None
        kwargs_shape['shape_scale_size'] = None
        
        shape_color : str = np.random.choice( COLOR_LIST, size = 1)[0] # color is in words (e.g. 'red')
        kwargs_shape['shape_color'] = COLOR_DICT_WORD_2_BGR_CODE[shape_color]
        kwargs_shape['shape_thickness'] = np.random.choice(FILL_NOFILL_np, size = 1)[0] # 5 = # Line thickness of 5 px
        kwargs_shape['shape_name'] = np.random.choice(SHAPE_TYPE_SPACE, size=1)[0] # print(np.random.choice(prog_langs, size=10, replace=True, p=[0.3, 0.5, 0.0, 0.2]))
        
        
        if kwargs_shape['shape_name'] == "Ellipse":
            kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
            kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
            kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] # default 0
        elif kwargs_shape['shape_name'] == "Parallelogram":
            kwargs_shape['a'] = np.random.choice(a_CENTER_SPACE_np, size = 1)[0]
            kwargs_shape['b'] = np.random.choice(b_CENTER_SPACE_np, size = 1)[0]
            kwargs_shape['alpha'] = np.random.choice(alpha_CENTER_SPACE_np, size = 1)[0] # default 90
        else:
            assert(False, 'The shape must be Ellipse or Parallelogram!')
        
        # fill in pandas df
        for c in COLUMNS:
            all_shapes_variable_data[c].append(kwargs_shape[c])
        
        new_shape = list_of_shapes.create_and_add_shape(kwargs_shape)
        first_generated_image.add_shape(shape=new_shape)
        #first_generated_image.add_shape_from_list(index_=i, list_of_shapes=list_of_shapes)

    first_generated_image.draw_grid_on_image(X_coors=X_CENTER_SPACE_np, 
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

if GENERATE_STATS:
    DF_all_shapes_variable_data = pd.read_csv(CSV_FILE_PATH)
    fig, axs = plt.subplots(3, 3, figsize=(20, 10))

    # create histograms
    for i_col, col in enumerate(COLUMNS):
        
        col_space = None
        bar_width = None
        
        if col == 'shape_color':
            # colors (and names) # string values
            df_color_hist = DF_all_shapes_variable_data[['shape_color_word']].apply(pd.value_counts).reset_index()
            #df_color_hist['shape_color_word'] = df_color_hist.apply(lambda row:  COLOR_DICT_BGR_2_WORD_CODE['-'.join([str(x) for x in row['index']])] , axis = 1)#.to_frame(name = 'shape_color_word')
            #df_color_hist.drop(['index'], axis=1, inplace=True)
            df_color_hist.set_index('shape_color_word', inplace=True)
            df_color_hist = df_color_hist.T.reset_index().drop(['index'], axis=1)
            
            colors_count = df_color_hist.columns
            colors_names = df_color_hist.values[0]
            colors_number = df_color_hist.shape[1]
            

            #plt.figure(i_col)
            axs[0, 1].set_title('Histogram of Shape Colors')
            axs[0, 1].bar(np.arange(colors_number), colors_count, align='center', width=0.5, color=colors_names, edgecolor='black')
            axs[0, 1].set_xticks(np.arange(colors_number), colors_names, size='small')
            axs[0, 1].set_yticks(np.arange(1, 1 + np.max(colors_count), HIST_Y_TICKS_STEP_SIZE), np.arange(1, 1 + np.max(colors_count), HIST_Y_TICKS_STEP_SIZE), size='small')
            axs[0, 1].grid(which='both')
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
        axs[i, j].set_yticks(np.arange(1, 1 + np.max(df_column_hist.values[0]), HIST_Y_TICKS_STEP_SIZE), np.arange(1, 1 + np.max(df_column_hist.values[0]), HIST_Y_TICKS_STEP_SIZE), size='small')
        axs[i, j].grid(which='both')
        axs[i, j].set_xlabel(xlabel)
        axs[i, j].set_ylabel(ylabel)
        #plt.savefig(f'DATA/{col}_stats.png')

    # delete unused subplots
    fig.delaxes(axs[0,2])
    fig.delaxes(axs[2,2])
    plt.tight_layout()
    plt.savefig(STATS_FILE_PATH)

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
