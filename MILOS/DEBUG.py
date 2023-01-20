from IPython.display import clear_output
from run_pipeline import run
import torch
import numpy as np
import imageio
clear_output(wait=True)
#clear_output(wait=True)
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
                                                                                                            image_id_batch,#id
                                                                                                            new_encoding_indices, 
                                                                                                            index_,
                                                                                                            dataset_str = dataset_type,
                                                                                                            create_plot_for_every_image_in_dataset = False, 
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
                    
trainers = {}
config_paths = {"toy_dataset" : "/home/novakovm/iris/MILOS/toy_shapes_config.yaml",
                "crafter_dataset" : "/home/novakovm/iris/MILOS/crafter_config.yaml"}
# toy dataset
trainers['toy_dataset'] = run(config_paths["toy_dataset"])
change_one_token(trainers, dataset_name = "toy_dataset", dataset_type = "test")
# crafter
trainers['crafter_dataset'] = run(config_paths["crafter_dataset"])
change_one_token(trainers, dataset_name = "crafter_dataset", dataset_type = "test")