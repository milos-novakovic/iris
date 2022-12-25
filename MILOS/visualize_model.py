from torchview import draw_graph
import graphviz
#K,D,run_id,M = 2**20, 256, 999, 1
#C,H,W = 3,64,64

# run following command in the file : /home/novakovm/iris/MILOS/
# python visualize_model.py 64 256 999 1

model = vq_vae_implemented_model #nn.Model
graph_name = "VQ_VAE"
save_graph = True
filename = f"001_VQ_VAE_K_64_D_256_M_1_bits_24"
directory = ""





# draw_graph
vq_vae_implemented_model_graph = draw_graph(model = model, 
                                            input_size=(64,C,H,W), 
                                            graph_name = graph_name,
                                            expand_nested=True,
                                            hide_module_functions = False,
                                            hide_inner_tensors = False,
                                            roll = True,
                                            save_graph = save_graph,
                                            filename = filename,
                                            directory = directory)
# visualize graph
# vq_vae_implemented_model_graph.visual_graph