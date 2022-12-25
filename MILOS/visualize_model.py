from torchview import draw_graph
#from torchvision.models import resnet18, GoogLeNet, densenet, vit_b_16
import graphviz
#K,D,run_id,M = 2**20, 256, 999, 1
#! pip install -q torchview
#! pip install -q -U graphviz

model = vq_vae_implemented_model #nn.Model
graph_name = "VQ_VAE"
save_graph = False
filename = ""
directory = ""


# when running on VSCode run the below command
# svg format on vscode does not give desired result
graphviz.set_jupyter_format('png')

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