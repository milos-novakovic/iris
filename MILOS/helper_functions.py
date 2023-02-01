import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio
import torchvision
import os
import glob

def get_hyperparam_from_config_file(config_file_path, hyperparam_name):
    #config_path = "/home/novakovm/iris/MILOS/toy_shapes_config.yaml"
    # Open the file and load the file
    with open(config_file_path) as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    for sub_config_dict in config_dict['training_hyperparams']:
        if hyperparam_name in sub_config_dict:
            return sub_config_dict[hyperparam_name]
    
    assert(False, f"There is no {hyperparam_name} at the config file at = {config_file_path}.")
    

### Vanilla Autoencoder params begin
def same_padding(h_in, w_in, s, k):
    # SAME padding: This is kind of tricky to understand in the first place because we have to consider two conditions separately as mentioned in the official docs.

    # Let's take input as n_i , output as n_o, padding as p_i, stride as s and kernel size as k (only a single dimension is considered)

    # Case 01: n_i \mod s = 0 :p_i = max(k-s ,0)

    # Case 02: n_i \mod s \neq 0 : p_i = max(k - (n_i\mod s)), 0)

    # p_i is calculated such that the minimum value which can be taken for padding. Since value of p_i is known, value of n_0 can be found using this formula (n_i - k + 2p_i)/2 + 1 = n_0.
    
    #SAME: Apply padding to input (if needed) so that input image gets fully covered by filter and stride you specified. For stride 1, this will ensure that output image size is same as input.
    # p = None
    # if n_i % s == 0:
    #     p = max(k-s ,0)
    # else:
    #     p = max((k - (n_i % s)), 0)
    # return p

    # n_out = np.ceil(float(n_in) / float(s))
    # p = None#int(max((n_out - 1) * s + k - n_in, 0) // 2)
    
    
    # if (n_in % s == 0):
    #     p = max(k - s, 0)
    # else:
    #     p = max(k - (n_in % s), 0)
    # #(2*(output-1) - input - kernel) / stride
    # p = int(p // 2)

    #out_height = np.ceil(float(h_in) / float(s[0]))
    #out_width  = np.ceil(float(h_in) / float(s[1]))

    #The total padding applied along the height and width is computed as:

    if (h_in % s[0] == 0):
        pad_along_height = max(k[0] - s[0], 0)
    else:
        pad_along_height = max(k[0] - (h_in % s[0]), 0)
        
    if (w_in % s[1] == 0):
        pad_along_width = max(k[1] - s[1], 0)
    else:
        pad_along_width = max(k[1] - (w_in % s[1]), 0)
  
    #Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_top, pad_bottom, pad_left, pad_right
# Hyper parameters

# EMPTY CACHE BEFORE RUNNING!
#torch.cuda.empty_cache()
def report_cuda_memory_status():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n" + f"Total Memory = {t/1e9 : .2f} GB.")
    print(f"Reserved Memory = {r/1e9 : .2f} GB.")
    print(f"Allocated inside Reserved Memory = {a/1e9 : .2f} GB.")
    print(f"Free inside Reserved Memory = {f/1e9 : .2f} GB.\n")

    global_free, total_gpu_memory_occupied = torch.cuda.mem_get_info('cuda:0') #for a given device 'cuda:' using cudaMemGetInfo  
    print(f'Global Free memory on the cuda:0 = {global_free/1e9 : .2f} GB.')
    print(f'Total GPU Memory occupied on the cuda:0 = {total_gpu_memory_occupied/1e9 : .2f} GB.\n')

# get the number of parameters
#from prettytable import PrettyTable

def count_parameters(model, path_to_write = '/home/novakovm/iris/MILOS/log_all.txt'):
    #table = PrettyTable(["Modules", "Parameters"])
    with open(path_to_write, 'a') as f:
        f.write(f"\n PyTorch print of the model:\n")
        f.write(f"\n{str(model)}\n\n")
        
    Modules_list = []
    Parameters_list = []
    Parameters_list_percent = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if (not parameter.requires_grad) or name[len(name)-4:] == 'bias':
            continue
        params = parameter.numel()
        params = int(params/1e3)
        Modules_list.append(name)
        Parameters_list.append(params)
        #table.add_row([name, params])
        total_params+=params
    Parameters_list_percent = [round(param_ / total_params * 100.,2) for param_ in Parameters_list]
    table = pd.DataFrame({"Module Name" : Modules_list, "# of params in thousands" : Parameters_list, "# of params [%]" : Parameters_list_percent})
    #print(table)
    print(f"Total Trainable Params in thousands: {total_params}")
    
    with open(path_to_write, 'a') as f:
        f.write(f"\n Total Trainable Params in thousands: {total_params} \n")
        f.write(f"\n{table.to_string()}\n\n")
    
    return total_params

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x, compose_transforms = None):
    # x dim = (N,C,H,W)
    if compose_transforms == None:
        return x
    
    #np.save('./DATA/RGB_mean.npy', RGB_mean) 
    #RGB_mean = np.load('./DATA/RGB_mean.npy')
    #RGB_std = np.load('./DATA/RGB_std.npy')
    RGB_mean = compose_transforms.transforms[0].mean
    RGB_std = compose_transforms.transforms[0].std
    
    R_mean, G_mean, B_mean = RGB_mean[0], RGB_mean[1], RGB_mean[2]
    R_std, G_std, B_std = RGB_std[0], RGB_std[1], RGB_std[2]
    
    MIN_PIXEL_VALUE, MAX_PIXEL_VALUE = 0,255
    # red chanel of the image
    x[:, 0, :, :] =  R_std * x[:, 0, :, :] + R_mean
    x[:, 0, :, :] = x[:, 0, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    # green chanel of the image
    x[:, 1, :, :] =  G_std * x[:, 1, :, :] + G_mean
    x[:, 1, :, :] = x[:, 1, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    # blue chanel of the image
    x[:, 2, :, :] =  B_std * x[:, 2, :, :] + B_mean
    x[:, 2, :, :] = x[:, 2, :, :].clamp(MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
    
    x = np.round(x) #x = np.round(x*255.)
    x = x.int()#astype(int)
    return x




# show/plot top-N worst reconstructed images
def show(original_imgs, reconstructed_imgs, imgs_ids, imgs_losses, savefig_path):
    N,C,H,W = original_imgs.size()
    fig = plt.figure(figsize=(2. * N, 1. * N))#rows,cols
    ncols = 8
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(N//4, ncols),  # creates 2x2 grid of axes
                    axes_pad=0.5,  # pad between axes in inch.
                    )    
    for i,axs in enumerate(grid):  
        im, original_or_reconstructed = None, None
        if i % 2 == 0:
            #axs.set_title(f"({i//2 + 1}/{N}) Org. Test img \n id={imgs_ids[i//2]} loss={imgs_losses[i//2]*1e3:.2f} e-3")
            im = original_imgs[i//2]
            original_or_reconstructed = "Org."
        else:
            #axs.set_title(f"({i//2 + 1}/{N}) Rec. Test img \n id={imgs_ids[i//2]} loss={imgs_losses[i//2]*1e3:.2f} e-3")
            im =  reconstructed_imgs[(i-1)//2]
            original_or_reconstructed = "Rec."
        
        # set title according to the image being original or reconstructed from the Test set (with test image id and its reconstruction loss)
        axs.set_title(f"({i//2 + 1}/{N}) {original_or_reconstructed} Test img \n id={imgs_ids[i//2]} loss={imgs_losses[i//2]*1e3:.2f} e-3")
        axs.imshow(np.transpose(im, (1, 2, 0))) # H,W,C
    
    # save figure and close plotter
    plt.savefig(savefig_path,bbox_inches='tight')
    plt.close()

def visualise_output(images, model, compose_transforms, imgs_ids, imgs_losses, savefig_path, device):

    with torch.no_grad():
        # original images
        # put original mini-batch of images to cpu
        original_images = images.to('cpu') # torch.Size([50, 3, 64, 64])
        
        # reconstructed images
        reconstructed_images = images.to(device)
        model = model.to(device)
        reconstructed_images = model(reconstructed_images)
        if len(reconstructed_images) == 5:
            reconstructed_images = reconstructed_images[1]
        
        # put reconstructed mini-batch of images to cpu
        reconstructed_images = reconstructed_images.to('cpu') # torch.Size([50, 3, 64, 64])        
        
        # diff_0_1 = (original_images-reconstructed_images)
        # # print statics on difference between original and reconstructed images (0.0-1.0 float range)
        # print("The test set difference between original and reconstructed images stats (0.0-1.0 float range):")
        # print(f"Size of tensor = {diff_0_1.size()}")
        # print(f"Mean of tensor = {diff_0_1.mean()}")
        # print(f"Min of tensor = {diff_0_1.min()}")
        # print(f"Max of tensor = {diff_0_1.max()}\n")
        
        original_images = to_img(original_images, compose_transforms)
        reconstructed_images = to_img(reconstructed_images, compose_transforms)        
        
        # diff_0_255 = (original_images-reconstructed_images)
        # # print statics on difference between original and reconstructed images (0-255 int range)
        # print("The test set difference between original and reconstructed images stats (0-255 int range):")
        # print(f"Size of tensor = {diff_0_255.size()}")
        # print(f"Mean of tensor = {diff_0_255.float().mean()}")
        # print(f"Min of tensor = {diff_0_255.min()}")
        # print(f"Max of tensor = {diff_0_255.max()}\n")
        
        #images = to_img(images, compose_transforms = compose_transforms)
        
        #np_imagegrid_original_images = torchvision.utils.make_grid(tensor = original_images, nrow = 10, padding = 5, pad_value = 255).numpy()
        #np_imagegrid_reconstructed_images = torchvision.utils.make_grid(tensor = reconstructed_images, nrow = 10, padding = 5, pad_value = 255).numpy()
        
        #fig, axs = plt.subplots(1, 10, figsize=(20, 10))
        #plt.imshow(np.transpose(np_imagegrid_original_images, (1, 2, 0))) # H,W,C
        #show(original_images,imgs_ids, imgs_losses)
        
        
        show(original_images,reconstructed_images,imgs_ids, imgs_losses, savefig_path)
        #plt.savefig('./SHOW_IMAGES/org_vs_rec_test_imgs.png',bbox_inches='tight')
        #plt.close()
        
        #fig, axs = plt.subplots(1, 10, figsize=(20, 10))
        #plt.imshow(np.transpose(np_imagegrid_reconstructed_images, (1, 2, 0))) # H,W,C
        #show(reconstructed_images,imgs_ids, imgs_losses)
        
        #show(original_images,reconstructed_images,imgs_ids, imgs_losses)
        #plt.savefig('./SHOW_IMAGES/autoencoder_output_test_50_images.png',bbox_inches='tight')
        #plt.close()


                    
def update_yaml(yaml_folder_name = "/home/novakovm/iris/MILOS/",
                yaml_file_name = "toy_shapes_config",
                get_new_data_from_human_readable_yaml_file = False,
                key = "NUM_WORKERS", 
                value = 5,
                value_type = int,
                source = "training_hyperparams"):
    # make a new yaml config file from human readable version of that same file
    yaml_full_dsc_path = yaml_folder_name + yaml_file_name +".yaml"
    if get_new_data_from_human_readable_yaml_file:
        yaml_full_src_path = yaml_folder_name + yaml_file_name+ "_human_readable" +".yaml"
        data_dict = yaml.load(open(yaml_full_src_path, 'r'), Loader=yaml.FullLoader)
        with open(yaml_full_dsc_path, 'w') as yaml_file:
            yaml_file.write( yaml.dump(data_dict, default_flow_style=False))
    
    if key != None and value != None:
        # update the newly created non-human readable file
        data_dict = yaml.load(open(yaml_full_dsc_path, 'r'), Loader=yaml.FullLoader)
        # pick the desired source list in the data dict. yaml file
        data_source_list = data_dict[source]
        # find the index in the list where the desired key is
        data_source_list_key_index = [idx for idx, data_key in enumerate(data_source_list) if key in data_key][0]
        # update step
        data_source_list[data_source_list_key_index][key] = value_type(value)
        # save the change
        with open(yaml_full_dsc_path, 'w') as yaml_file:
            yaml_file.write( yaml.dump(data_dict, default_flow_style=False))
            


def Grid_of_Randomly_Selected_images_for_MSc_thesis_writing(dataset_short_name = "two-shapes", nrows = 4, ncols = 12, zoom_scale_factor = 1.):#"one-shape" # "crafter"
    #[Grid_of_Randomly_Selected_images_for_MSc_thesis_writing(dataset_short_name) for dataset_short_name in ["one-shape", "two-shapes", "crafter"]]
     # 16th images in total shown
    H,W = 64,64 # height and width of the loaded images
    
    dataset_full_name = {}
    dataset_full_name["one-shape"]  = f"The One-shape Dataset {nrows}x{ncols}"
    dataset_full_name["two-shapes"] = f"The Two-shapes Dataset {nrows}x{ncols}"
    dataset_full_name["crafter"]    = f"The Crafter Dataset {nrows}x{ncols}"

    train_data_file_path = {}
    train_data_file_path["one-shape"]   = "/home/novakovm/DATA_TRAIN_ONE_SHAPE/"# One-shape dataset
    train_data_file_path["two-shapes"]  = "/home/novakovm/DATA_TRAIN/"          # Two-shapes dataset
    train_data_file_path["crafter"]     = "/home/novakovm/crafter/DATA_TRAIN/"  # Crafter dataset

    TOTAL_NUMBER_OF_IMAGES = {}
    TOTAL_NUMBER_OF_IMAGES["one-shape"] = 16_384      # One-shape dataset
    TOTAL_NUMBER_OF_IMAGES["two-shapes"]= 268_435_456 # Two-shapes dataset
    TOTAL_NUMBER_OF_IMAGES["crafter"]   = 900_000     # Crafter dataset

    image_id = 1
    images_data = []
    np.random.seed(41)#for reproducibility
    random_start_image_number = np.random.choice(1000)
    
    for image_file_path in glob.iglob(train_data_file_path[dataset_short_name] + '*.png'):
        if random_start_image_number > 0:
            random_start_image_number -=1
            continue
        
        if image_id > (ncols*nrows):
            break
        image_id += 1
        #image_file_path = train_data_file_path[dataset_short_name] + 'color_img_' + str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES[dataset_short_name]))) + '.png'
        #print(image_file_path)
        if os.path.isfile(image_file_path): 
            image = torchvision.io.read_image(image_file_path, mode=torchvision.io.image.ImageReadMode.RGB) # .double() = torch.float64 and  .float() = torch.float32
            image = image.permute(1,2,0)
            images_data.append(image)
            print(f"Loading {len(images_data)}th image completed!")
            #print(f"Img. min = {image.min()}, Img. max = {image.max()}, Img. mean = {image.mean()}")
        else:
            print(f"Sorry the image {'color_img_' + str(image_id).zfill(len(str(TOTAL_NUMBER_OF_IMAGES[dataset_short_name]))) + '.png'} is not in the training set.")
    
    reduce_left_and_right_margin = 2 if ncols >= 4 else 0
    fig = plt.figure(figsize=(zoom_scale_factor * (float(ncols)-reduce_left_and_right_margin), zoom_scale_factor * float(nrows)))


    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                    axes_pad=0.2,  # pad between axes in inch.
                    share_all=True, # all axes share their x- and y-axis.
                    label_mode = "L" # all axes are labelled.
                    )

    for ax, im in zip(grid, images_data):    
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        
    grid[0].set_xticks([0, W - 1])
    grid[0].set_yticks([0, H - 1])
    
    fig.suptitle(f"{nrows} x {ncols} grid of randomly selected\nimages from {dataset_full_name[dataset_short_name]}", fontsize=10)
    plt.tight_layout()
    plt.show()

    SAVING_FOLDER_PATH = "/home/novakovm/Grid_of_Randomly_Selected_images_for_MSc_thesis_writing/"
    if not(os.path.isdir(SAVING_FOLDER_PATH)):
        os.mkdir(SAVING_FOLDER_PATH)

    fig.savefig(SAVING_FOLDER_PATH + dataset_full_name[dataset_short_name] + ".png")