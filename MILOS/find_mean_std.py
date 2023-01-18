import numpy as np
import cv2
def find_mean_std(TOTAL_NUMBER_OF_IMAGES, 
                  image_ids_numbers = None, 
                  method = 'n',
                  NUMPY_DOUBLE_CHECK= True,
                  train_folder_path = '/home/novakovm/DATA_TRAIN/',
                  main_folder_path = '/home/novakovm/iris/MILOS/'
                  ):
    
    # method = 'n' or method = 'n-1'
    
    # image_ids_numbers is the list of numbers for what images to produce estimate of mean and variance
    # (so that it works for full batch, i.e. all training images, when image_ids_numbers = None)
    # and for a subset of images when image_ids_numbers has acctually elements
    if image_ids_numbers is None:
        # full batch
        image_ids = [str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES))) for image_id in range(TOTAL_NUMBER_OF_IMAGES)]
    else:
        # mini-batch
        image_ids = [str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES))) for image_id in image_ids_numbers]

    #empirical calculation of first centered moment per channel of image
    x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue = \
    0.,0.,0.#np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT)

    #empirical calculation of second centered moment per channel of image
    x2_mean_bar_ch2_red, x2_mean_bar_ch1_green, x2_mean_bar_ch0_blue = \
    0.,0.,0.#np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT),  np.zeros((H,W), dtype=FLOAT)

    # only red, green, and blue channel lists for all generated images
    R_imgs,G_imgs,B_imgs = [],[],[]
    
    # estimate E[X] and E[X^2] across all chanells and not for a specific chanell
    X_mean_bar_all_ch, X2_mean_bar_all_ch = 0., 0.

    for i,image_id in enumerate(image_ids):
        if i % 10000 == 0:
            print(f"{i}/{len(image_ids)} image preprocessed! ({i / len(image_ids) * 100 : .1f}%)")
        
        # update empirical estimates for 1st and 2nd centered moments        
        image_full_path = train_folder_path + 'color_img_' + image_id + '.png'
        
        x = cv2.imread(image_full_path)
        
        if NUMPY_DOUBLE_CHECK:
            R_imgs.append(np.float64(x)[:,:,2])
            G_imgs.append(np.float64(x)[:,:,1])
            B_imgs.append(np.float64(x)[:,:,0])

        # empirically and recursively calculate first moment (E[X] = expectation) per chanel (red, green, blue)
        x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue, X_mean_bar_all_ch = \
            x_mean_bar_ch2_red   + (-x_mean_bar_ch2_red   + np.mean(np.float64(x)[:,:,2]))/(i+1), \
            x_mean_bar_ch1_green + (-x_mean_bar_ch1_green + np.mean(np.float64(x)[:,:,1]))/(i+1), \
            x_mean_bar_ch0_blue  + (-x_mean_bar_ch0_blue  + np.mean(np.float64(x)[:,:,0]))/(i+1), \
            X_mean_bar_all_ch    + (-X_mean_bar_all_ch    + np.mean(np.float64(x)[:,:,:]))/(i+1)
        
        # empirically and recursively calculate second centered moment (E[X^2] = VAR[X] + E[X]^2 = std(X)^2 + E[X]^2) per chanel (red, green, blue)
        x2_mean_bar_ch2_red, x2_mean_bar_ch1_green, x2_mean_bar_ch0_blue, X2_mean_bar_all_ch = \
            x2_mean_bar_ch2_red     + (-x2_mean_bar_ch2_red   + np.mean(np.float64(x)[:,:,2]**2))/(i+1), \
            x2_mean_bar_ch1_green   + (-x2_mean_bar_ch1_green + np.mean(np.float64(x)[:,:,1]**2))/(i+1), \
            x2_mean_bar_ch0_blue    + (-x2_mean_bar_ch0_blue  + np.mean(np.float64(x)[:,:,0]**2))/(i+1), \
            X2_mean_bar_all_ch      + (-X2_mean_bar_all_ch    + np.mean(np.float64(x)[:,:,:]**2))/(i+1)

    #available
    #x_mean_bar_ch2_red, x_mean_bar_ch1_green, x_mean_bar_ch0_blue, X_mean_bar_all_ch
    x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue, X_std_bar_all_ch = None, None, None, None
    
    if method == 'n-1': # unbiased
        x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue, X_std_bar_all_ch = \
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch2_red   - x_mean_bar_ch2_red**2)),  \
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch1_green - x_mean_bar_ch1_green**2)),\
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (x2_mean_bar_ch0_blue  - x_mean_bar_ch0_blue**2)), \
            np.sqrt((TOTAL_NUMBER_OF_IMAGES*1.) / (TOTAL_NUMBER_OF_IMAGES - 1.) * (X2_mean_bar_all_ch    - X_mean_bar_all_ch**2))
              

    elif method == 'n': # numpy
        x_std_bar_ch2_red, x_std_bar_ch1_green, x_std_bar_ch0_blue, X_std_bar_all_ch = \
            np.sqrt(x2_mean_bar_ch2_red   - x_mean_bar_ch2_red**2),  \
            np.sqrt(x2_mean_bar_ch1_green - x_mean_bar_ch1_green**2),  \
            np.sqrt(x2_mean_bar_ch0_blue  - x_mean_bar_ch0_blue**2), \
            np.sqrt(X2_mean_bar_all_ch  - X_mean_bar_all_ch**2)
    else:
        assert(False)
    

    # two arrays of (3,) size
    RGB_mean = np.array([x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue])#np.dstack((x_mean_bar_ch2_red,x_mean_bar_ch1_green,x_mean_bar_ch0_blue))
    RGB_std = np.array([x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue])#np.dstack((x_std_bar_ch2_red,x_std_bar_ch1_green,x_std_bar_ch0_blue))    
    
    # cast to np.array
    X_mean_bar_all_ch = np.array(X_mean_bar_all_ch) 
    X_std_bar_all_ch = np.array(X_std_bar_all_ch)
    
    if NUMPY_DOUBLE_CHECK:
        R_imgs,G_imgs,B_imgs = np.array(R_imgs),np.array(G_imgs),np.array(B_imgs)
        RGB_mean_np = np.array([np.mean(R_imgs),np.mean(G_imgs),np.mean(B_imgs)])
        RGB_std_np = np.array([np.std(R_imgs),np.std(G_imgs),np.std(B_imgs)])
        
        Total_mean_np = np.mean(np.array([R_imgs, G_imgs, B_imgs]).reshape(-1))
        Total_std_np = np.std(np.array([R_imgs, G_imgs, B_imgs]).reshape(-1))
    
    # save preprocessing result (i.e. mean and std per changel)
    
    np.save(main_folder_path + 'RGB_mean.npy', RGB_mean) # np.load('./DATA/RGB_mean.npy')
    np.save(main_folder_path + 'RGB_std.npy', RGB_std) # np.load('./DATA/RGB_std.npy')
    
    np.save(main_folder_path + 'train_dataset_mean.npy', X_mean_bar_all_ch) # np.load('./DATA/RGB_mean.npy')
    np.save(main_folder_path + 'train_dataset_std.npy', X_std_bar_all_ch) # np.load('./DATA/RGB_std.npy')
    
    print(f"Training Images Mean per chanel= {np.round(RGB_mean,2)}")
    print(f"Training Images Std per chanel = {np.round(RGB_std,2)}")
    print(f"Training Images Mean           = {np.round(X_mean_bar_all_ch,2)}")
    print(f"Training Images Std            = {np.round(X_std_bar_all_ch,2)}\n")   
    
    if NUMPY_DOUBLE_CHECK:
        np.save(main_folder_path + 'RGB_mean_np.npy', RGB_mean_np) # np.load('./DATA/RGB_mean_np.npy')
        np.save(main_folder_path + 'RGB_std_np.npy', RGB_std_np) # np.load('./RGB_std_np/RGB_mean.npy')
        print(f"Mean (Empirical - np) diff per chanel= {RGB_mean - RGB_mean_np}")
        print(f"Std (Empirical - np) diff per chanel = {RGB_std - RGB_std_np}")
        print(f"Mean (Empirical - np) diff           = {X_mean_bar_all_ch - Total_mean_np}")
        print(f"Std (Empirical - np) diff            = {X_std_bar_all_ch - Total_std_np}")
    
    
    #np.sum(np.abs(RGB_mean - RGB_mean_np)) < 1e-14 #  True
    #np.sum(np.abs(RGB_std - RGB_std_np)) < 1e-13 #  True
    
    if NUMPY_DOUBLE_CHECK:
        return RGB_mean, RGB_std, RGB_mean_np, RGB_std_np, X_mean_bar_all_ch, X_std_bar_all_ch,  Total_mean_np, Total_std_np
    else:
        return RGB_mean, RGB_std, None, None, X_mean_bar_all_ch, X_std_bar_all_ch, None, None

