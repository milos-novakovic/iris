import torch
import torch.nn as nn
import torch.nn.functional as F
class VectorQuantizer(nn.Module):
    def __init__(self, args_VQ):
        super(VectorQuantizer, self).__init__()
        
        self.train_with_quantization = args_VQ['train_with_quantization']
        self.D = args_VQ['D']
        self.K = args_VQ['K']
        self.beta = args_VQ['beta']
        self.M = args_VQ['M']
        
        self.E = nn.Embedding(self.K, self.D)
        self.E.weight.data.uniform_(-1/self.K, 1/self.K)
        
        
        #self.token_usage = torch.zeros(size = (self.M+1, self.M+1)) # (M+1) x (M+1) matrix, where each element is the number of that particular (row,col)-position token used
        
        # Try this
        # taken from "Vector-quantized Image Modeling with Improved VQGAN" paper from ICLR (2022) by Jiahui Yu et al.
        
        # self.codebook_distribution_initialization = args_VQ['codebook_distribution_initialization']
        # if self.codebook_distribution_initialization == 'uniform':
        #    self.E.weight.data.uniform_(-1/self.K, 1/self.K)
        # elif self.codebook_distribution_initialization == 'standard_normal':
        #    self.E.weight.data.normal_()
        #
        
        #self.requires_normalization_with_sphere_projection = args_VQ['requires_normalization_with_sphere_projection']
        
        
        
        
        # specifices the usage of the VQ-VAE dictionary (codebook) E updates with Exponential Moving Average (EMA)
        self.use_EMA = args_VQ['use_EMA']
        

        
        if self.use_EMA:
            # decay in the Exp. Moving Average
            self.gamma = args_VQ['gamma']
            
            #register_buffer(.) functionality is typically used to register a buffer that should not to be considered a model parameter. For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state. # https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch
            #N_i^{(t)} - notation used in the paper "Neural Distance Representation Learning"
            self.register_buffer('N_ema', torch.zeros(self.K)) 
            #self.register_buffer('N_ema', torch.zeros(self.K).requires_grad_(False), persistent=False) 
            
            #m_i^{(t)} \in R^D - notation used in the paper "Neural Distance Representation Learning"
            self.M_ema = nn.Parameter(torch.Tensor(self.K, self.D))
            self.M_ema.data.normal_() 
            #self.register_buffer('M_ema', torch.Tensor(self.K, self.D).normal_().requires_grad_(False), persistent=False) 
            #self.register_buffer('M_ema', torch.empty(self.K, self.D).normal_().requires_grad_(True), persistent=False) 
            
            
    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # C = D
        Ze_tensor = inputs.permute(0, 2, 3, 1).contiguous()
        Ze_tensor_shape = Ze_tensor.shape
        
        # Flatten input = dims (B*H*W) x D
        Ze = Ze_tensor.view(-1, self.D)
                
        # add normalization (i.e. l2-sphere projection) F.normalize(x, dim=-1)
        # taken from "Vector-quantized Image Modeling with Improved VQGAN" paper from ICLR (2022) by Jiahui Yu et al.
        # if self.requires_normalization_with_sphere_projection:
        #   Ze = F.normalize(input = Ze, p = 2, dim = -1) # since Ze tensor is of shape BHWC we will use last chanell dimension (dim = -1) then all of the B*H*W of C-dim. non-quantized encoded latent vecotrs \vec{Z_e} is going to be normlized
        #   self.E.weight = F.normalize(input = self.E.weight, p = 2, dim = -1) # since Zq tensor is of shape BHWC we will use last chanell dimension (dim = -1) then all of the B*H*W of C-dim. quantized encoded latent vecotrs \vec{Z_q} is going to be normlized
        
        
        # Calculate distances matrix D = dims (B*H*W) x K
        D = (torch.sum(Ze**2, dim=1, keepdim=True) # sum across axis 1 matrix dims (but keepdims) = [(B*H*W) x D] -> column (B*H*W)-sized vector -> torch broadcast to same K-rows to get matrix = dims (B*H*W) x K
            + torch.sum(self.E.weight**2, dim=1) # sum across axis 1 matrix dims = [K x D] -> row K-sized vector -> torch broadcast to same (B*H*W)-columns to get matrix = dims (B*H*W) x K
            - 2 * torch.matmul(Ze, self.E.weight.t())) # [matrix dims = (B*H*W) x D] * [matrix dims = D x K]^T
            
        # Discretization metric is the closes vector in l2-norm sense
        # Encoding indices = dims (B*H*W) x 1
        encoding_indices = torch.argmin(D, dim=1).unsqueeze(1)
        
        # go into the debugger here
        #encoding_indices_per_position = encoding_indices.detach().view(Ze_tensor_shape[0], Ze_tensor_shape[1], Ze_tensor_shape[2])# size = B, W, H
        # encoding_indices_per_position_words, encoding_indices_per_position_freq = torch.unique(input = encoding_indices_per_position, sorted=True, return_inverse=False, return_counts=True, dim=0)
        #self.token_usage[encoding_indices_per_position_words] += encoding_indices_per_position_freq
        
        # from embedding matrix E (dims K x D) pick (B*H*W)-number of vectors 
        # put it into original shape of the tensor that VQ got from Encoder, i.e. Ze_tensor_shape
        Zq_tensor = self.E(encoding_indices).view(Ze_tensor_shape)# size = [B, H, W, C]
        
        
        # calculate the avg. estimated probably of the quantized index 1,..,K occuring
        # https://pytorch.org/docs/stable/generated/torch.unique.html 
        # https://stackoverflow.com/questions/10741346/frequency-counts-for-unique-values-in-a-numpy-array
        
        # torch.unique gives back two arguments = 
        # (sorted array of unique elements that are in the encoding_indices array) and
        # (number of occurances of each i-th element in the sorted array without duplicates in the encoding_indices array)
        estimate_codebook_words, estimate_codebook_words_freq = torch.unique(input = encoding_indices,#encoding_indices.detach(),
                                                                            sorted=True, 
                                                                            return_inverse=False,
                                                                            return_counts=True,
                                                                            dim=0) # size of K elements
        # Try to use EMA VQ-VAE dict. update algorithm
        if self.use_EMA:
            if self.training:
                # update the codebook words occurance counting freq. [K-sized array]
                current_N_ema_estimate = torch.zeros(self.N_ema.size(), device=self.N_ema.device)
                current_N_ema_estimate[estimate_codebook_words.view(-1)] = estimate_codebook_words_freq.view(-1).float()
                self.N_ema = self.gamma*self.N_ema + (1-self.gamma)* current_N_ema_estimate #estimate_codebook_words_freq
                
                # Laplace smoothing of the cluster size (in case there is a cluster i \in {1,...K} that isn't occuring at all in the N_ema array)
                # https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
                N = torch.sum(self.N_ema.data)
                self.N_ema = ((self.N_ema + 1e-5) / (N + self.K * 1e-5) * N)
                
                # update the online EMA-weighted average of closest vectors
                self.M_ema = nn.Parameter(self.gamma * self.M_ema + (1-self.gamma) *  torch.matmul(torch.zeros(encoding_indices.shape[0], self.K, device=inputs.device).scatter_(1, encoding_indices, 1).t(), Ze) )
                #self.M_ema = self.gamma * self.M_ema + (1-self.gamma) *  torch.matmul(torch.zeros(encoding_indices.shape[0], self.K, device=inputs.device).scatter_(1, encoding_indices.detach(), 1).t(), Ze.detach())

                # update the codebook vectors (online EMA-weighted average of the closest vectors divided by their number of occurances)
                self.E.weight = nn.Parameter(self.M_ema / self.N_ema.unsqueeze(1))
                
        # Encoder distance tensor to Quantized vectors Loss calculation
        e_latent_loss = F.mse_loss(Zq_tensor.detach(), Ze_tensor) # commitment loss
        
        # Quantized distance tensor to Encoder vectors Loss calculation (no loss term for updating the codebook discrete vectors if we are updating them already using EMA method)
        q_latent_loss = torch.tensor(0) if self.use_EMA else F.mse_loss(Zq_tensor,          Ze_tensor.detach())
        
        # Combination of two losses
        e_and_q_latent_loss = q_latent_loss + self.beta * e_latent_loss
    
        # in + (out - in).detach() 
        Zq_tensor = Ze_tensor + (Zq_tensor - Ze_tensor).detach()        
        
        # BHWC -> BCHW
        Zq = Zq_tensor.permute(0, 3, 1, 2).contiguous()
        
        # create a zero K-sized array of probabilitites associated of a word occuring (i.e. being used in the coding) [vector of size K]
        estimate_codebook_words_prob = torch.zeros(self.K, device=inputs.device)
        
        # update probabilitites with freq. of the occuring of the codebook words [vector of size K]
        estimate_codebook_words_prob[estimate_codebook_words.view(-1)] = estimate_codebook_words_freq.view(-1).float()
        
        # normalize probability so that it sums up to 1 [vector of size K]
        #estimate_codebook_words_prob = estimate_codebook_words_freq.detach() / torch.sum(estimate_codebook_words_freq.detach())  #not this
        estimate_codebook_words_prob = estimate_codebook_words_prob.detach() / torch.sum(estimate_codebook_words_prob.detach()) #use this
        
        # calculate the log2(.) of probabilitites [vector of size K]
        log_estimate_codebook_words_prob = torch.log2(estimate_codebook_words_prob + 1e-10)
        
        # estimate (calculate) the entropy of the codewords in bits (log2 base) [scalar value]
        estimate_codebook_words_entropy_bits = - torch.sum(estimate_codebook_words_prob * log_estimate_codebook_words_prob)
        
        # calculate the rest of estimators to estimate perplexity = exp(entropy of codewords inside codebook E) [scalar value]
        estimate_codebook_words = 2**(estimate_codebook_words_entropy_bits)
        
        return e_and_q_latent_loss, Zq, e_latent_loss.item(), q_latent_loss.item(), estimate_codebook_words.item()
    
class Residual(nn.Module):
    def __init__(self, res_block_args):
        super(Residual, self).__init__()
        C_in = res_block_args['C_in']
        C_mid =res_block_args['C_mid']
        
        self.residual_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=C_in,
                      out_channels = C_mid, #num_residual_hiddens,#32
                      kernel_size=3, 
                      stride=1,
                      padding=1,
                      bias=False), 
            # H & W stay the same, just number of ch. changes from C_in to C_mid
            nn.ReLU(True),
            nn.Conv2d(in_channels=C_mid,#num_residual_hiddens, #32
                      out_channels=C_in,#num_hiddens,#128
                      kernel_size=1,
                      stride=1,
                      bias=False)
            # H & W stay the same, just number of ch. changes from C_mid to C_in
        )
    
    def forward(self, x):
        # two CNN (resulution-agnostic) layers and input are simply added to create a residual path
        return x + self.residual_block(x)


class ResidualStack(nn.Module):
    def __init__(self, res_block_args):
        super(ResidualStack, self).__init__()
        
        # save residual block arguments in dict
        self.res_block_args = res_block_args
        
        # Add sequentially block_size number of Residual (two CNNs and a residual path) blocks
        self.residual_blocks = nn.ModuleList([Residual(res_block_args) for _ in range(self.res_block_args['block_size'])])

    def forward(self, x):
        # sequentially call each block one after the other; and do forward pass on each block
        for i in range(self.res_block_args['block_size']):
            x = self.residual_blocks[i](x)
            
        # because blocks finish with one CNN that does not have an activation function put
        # the activation function to be ReLU at the end
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, args_encoder, res_block_args):
        super(Encoder, self).__init__()

        # save Encoder arguments in dict
        self.args_encoder = args_encoder
        
        # Series of conv layers with stride = 2 to reduce the image resolution by a factor of 2 for each conv layer 
        self.sequential_convs = torch.nn.Sequential()
        # self.sequential_convs = nn.Sequential(
        #                                         nn.Conv2d(  in_channels=args_encoder['conv1_Cin'],#Cin
        #                                                     out_channels=args_encoder['conv1_Cout'],#64
        #                                                     kernel_size=args_encoder['conv1_k'],#4
        #                                                     stride=args_encoder['conv1_s'],#2
        #                                                     padding=args_encoder['conv1_p']),#1
        #                                         nn.ReLU(True),
        #                                         nn.Conv2d(  in_channels=args_encoder['conv2_Cin'],#64
        #                                                     out_channels=args_encoder['conv2_Cout'],#128
        #                                                     kernel_size=args_encoder['conv2_k'],#4
        #                                                     stride=args_encoder['conv2_s'],#2
        #                                                     padding=args_encoder['conv2_p']), #, #1
        #                                         nn.ReLU(True),
        #                                         nn.Conv2d(  in_channels=args_encoder['conv3_Cin'],#128
        #                                                     out_channels=args_encoder['conv3_Cout'],#128
        #                                                     kernel_size=args_encoder['conv3_k'],#4
        #                                                     stride=args_encoder['conv3_s'],#2
        #                                                     padding=args_encoder['conv3_p']),#1
        #                                         nn.ReLU(True),
        #                                         nn.Conv2d(  in_channels=args_encoder['conv4_Cin'],#128
        #                                                     out_channels=args_encoder['conv4_Cout'],#128
        #                                                     kernel_size=args_encoder['conv4_k'],#4
        #                                                     stride=args_encoder['conv4_s'],#2
        #                                                     padding=args_encoder['conv4_p'])#1
        #                                         #nn.ReLU(True)#input to the residual stack already has ReLU at the input, so we dont put it here (applying ReLU two times does not change anything but increases the computational time)
        #                                     )
        
        for i in range(1, 1 + args_encoder['encoder_conv_stride2_layer_number']):
            self.sequential_convs.add_module(f"conv2d_{i}", nn.Conv2d(  in_channels=args_encoder[f'conv{i}_Cin'],#64
                                                                        out_channels=args_encoder[f'conv{i}_Cout'],#3
                                                                        kernel_size=args_encoder[f'conv{i}_k'],#4
                                                                        stride=args_encoder[f'conv{i}_s'],#2
                                                                        padding=args_encoder[f'conv{i}_p']))
            self.sequential_convs.add_module(f"ReLU_{i}", nn.ReLU(True))

        # Do not add ReLU() activation function layer to the channel_adjusting_conv CNN
        # input to the residual stack already has ReLU at the input, 
        # so we dont put it here (applying ReLU two times does not change anything but increases the computational time)
        self.pre_residual_stack = nn.Conv2d(in_channels=args_encoder['pre_residual_stack_conv_Cin'], #128 
                                                out_channels=args_encoder['pre_residual_stack_conv_Cout'], #128
                                                kernel_size=args_encoder['pre_residual_stack_conv_k'],#3
                                                stride=args_encoder['pre_residual_stack_conv_s'],#1
                                                padding=args_encoder['pre_residual_stack_conv_p'])#3
        
        # Series of two-depth residual blocks packed into a residual stack (e.g. ResNet)
        self.residual_stack = ResidualStack(res_block_args) 
        
        self.channel_adjusting_conv = nn.Conv2d(in_channels=args_encoder['channel_adjusting_conv_Cin'], #128
                                                out_channels=args_encoder['channel_adjusting_conv_Cout'], #D,
                                                kernel_size=args_encoder['channel_adjusting_conv_k'],#1
                                                stride=args_encoder['channel_adjusting_conv_s'],#1
                                                padding=args_encoder['channel_adjusting_conv_p'])#0
        
        # Try this
        # taken from "Vector-quantized Image Modeling with Improved VQGAN" paper from ICLR (2022) by Jiahui Yu et al.
        # self.requires_projection = args_encoder['D'] != args_encoder['D_e']
        # if self.requires_projection:
            # self.D_e = args_encoder['D_e']
            # self.D = args_encoder['D']
            # self.project_in = nn.Linear(in_features = self.D_e, out_features = self.D, bias=True, device=None, dtype=None)# if self.requires_projection else nn.Identity() 
        
        
        #if args_encoder['M'] == 0:
            # First we calculate the padding
            #padding = (0,1,0,1) #['W_left'], ['Padding_W_right'], ['Padding_H_top'], ['Padding_H_bottom']
            #self.zeropad1=  nn.ZeroPad2d(padding)

    def forward(self, x):
        x = self.sequential_convs(x)
        x = self.pre_residual_stack(x)
        x = self.residual_stack(x)
        x = self.channel_adjusting_conv(x)
        
        # Try this
        # taken from "Vector-quantized Image Modeling with Improved VQGAN" paper from ICLR (2022) by Jiahui Yu et al.
        # if self.requires_projection:
            # B_e, _, H_e, W_e = x.shape
            # x = self.project_in(x.view(-1, self.D_e))
            # x = x.view(B_e, self.D, H_e, W_e)
        
        #if self.args_encoder['M'] == 0:
            #x = self.zeropad1(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, args_decoder, res_block_args):
        super(Decoder, self).__init__()
        
        
        
        # Try this
        # taken from "Vector-quantized Image Modeling with Improved VQGAN" paper from ICLR (2022) by Jiahui Yu et al.
        # self.requires_projection = args_decoder['D'] != args_decoder['D_e']
        # if self.requires_projection:
            # self.D_e = args_decoder['D_e']
            # self.D = args_decoder['D']
            # self.project_out = nn.Linear(in_features = self.D, out_features = self.D_e, bias=True, device=None, dtype=None)# if self.requires_projection else nn.Identity() 
        
        
        
        # Do not add ReLU() activation function layer to the channel_adjusting_conv CNN
        # input to the residual stack already has ReLU at the input, 
        # so we dont put it here (applying ReLU two times does not change anything but increases the computational time)
        self.channel_adjusting_conv = nn.Conv2d(in_channels=args_decoder['channel_adjusting_conv_Cin'],#D
                                                out_channels=args_decoder['channel_adjusting_conv_Cout'],#128
                                                kernel_size=args_decoder['channel_adjusting_conv_k'],#3 ; TRY WITH KERNEL SIZE 1 AND PADDING 0
                                                stride=args_decoder['channel_adjusting_conv_s'],#1
                                                padding=args_decoder['channel_adjusting_conv_p'])#1 ; TRY WITH KERNEL SIZE 1 AND PADDING 0
        
        self.residual_stack = ResidualStack(res_block_args)
        
        self.sequential_trans_convs = torch.nn.Sequential()
        
        for i in range(1,1 + args_decoder['encoder_conv_stride2_layer_number']):
            self.sequential_trans_convs.add_module(f"conv2d_{i}", nn.ConvTranspose2d(   in_channels=args_decoder[f'trans_conv{i}_Cin'],#64
                                                                                        out_channels=args_decoder[f'trans_conv{i}_Cout'],#3
                                                                                        kernel_size=args_decoder[f'trans_conv{i}_k'],#4
                                                                                        stride=args_decoder[f'trans_conv{i}_s'],#2
                                                                                        padding=args_decoder[f'trans_conv{i}_p']))#1
            if i != args_decoder['encoder_conv_stride2_layer_number']:
                #Do not add ReLU() activation function layer to the last CNN
                self.sequential_trans_convs.add_module(f"ReLU_{i}", nn.ReLU(True))
        
        # i = args_decoder['CONV_LAYERS']+1
        # self.sequential_trans_convs.add_module(f"conv2d_{i}", nn.ConvTranspose2d(   in_channels=args_decoder[f'trans_conv{i}_Cin'],#64
        #                                                                             out_channels=args_decoder[f'trans_conv{i}_Cout'],#3
        #                                                                             kernel_size=args_decoder[f'trans_conv{i}_k'],#3
        #                                                                             stride=args_decoder[f'trans_conv{i}_s'],#1
        #                                                                             padding=args_decoder[f'trans_conv{i}_p']))#0

        
        # Manual fill in of the nn.Sequential() class wrapper of the CNN layers
        # self.sequential_trans_convs = nn.Sequential(
        #                                             nn.ConvTranspose2d( in_channels=args_decoder['trans_conv1_Cin'],#128
        #                                                                 out_channels=args_decoder['trans_conv1_Cout'],#64
        #                                                                 kernel_size=args_decoder['trans_conv1_k'],#4
        #                                                                 stride=args_decoder['trans_conv1_s'],#2
        #                                                                 padding=args_decoder['trans_conv1_p']),#1
        #                                             nn.ReLU(True),
        #                                             nn.ConvTranspose2d( in_channels=args_decoder['trans_conv2_Cin'],#64
        #                                                                 out_channels=args_decoder['trans_conv2_Cout'],#3
        #                                                                 kernel_size=args_decoder['trans_conv2_k'],#4
        #                                                                 stride=args_decoder['trans_conv2_s'],#2
        #                                                                 padding=args_decoder['trans_conv2_p'])#1
        #                                             )
    def forward(self, x):
        # Try this
        # taken from "Vector-quantized Image Modeling with Improved VQGAN" paper from ICLR (2022) by Jiahui Yu et al.
        # if self.requires_projection:
            # B_e, _, H_e, W_e = x.shape
            # x = self.project_out(x.view(-1, self.D))
            # x = x.view(B_e, self.D_e, H_e, W_e)
        
        x = self.channel_adjusting_conv(x)
        x = self.residual_stack(x)
        x = self.sequential_trans_convs(x)    
        return x

class VQ_VAE(nn.Module):
    def __init__(self, args_encoder, args_VQ, args_decoder, res_block_args_encoder, res_block_args_decoder):
        super(VQ_VAE, self).__init__()
        
        ######################
        # Model Constructors #
        ######################
        self.C_in, self.H_in, self.W_in = args_encoder['C_in'], args_encoder['H_in'], args_encoder['W_in']
        self.args_encoder=args_encoder
        self.args_VQ=args_VQ
        self.args_decoder=args_decoder
        self.res_block_args_encoder = res_block_args_encoder
        self.res_block_args_decoder = res_block_args_decoder
        
        self.train_with_quantization = args_VQ['train_with_quantization']
        
        self.encoder =  Encoder(self.args_encoder, self.res_block_args_encoder)
        
        if self.train_with_quantization:
            self.VQ      =  VectorQuantizer(args_VQ)
            
        self.decoder =  Decoder(self.args_decoder, self.res_block_args_decoder)

    def forward(self, x):                       #torch.Size([128, 3, 64, 64])
        Ze = self.encoder(x)                    #torch.Size([128, 64, 16, 16])
        
        e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = None, None, None, None, None
        if self.train_with_quantization:
            e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = self.VQ(Ze)   #torch.Size([128, 64, 16, 16])
        else:
            e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy =  0, Ze, 0, 0, 0
        # use this for simple countinous vanilla AE (i.e. to bypass the VQ class forward pass)
        #e_and_q_latent_loss, Zq,  e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy = 0, Ze, 0, 0, 0         #torch.Size([128, 64, 16, 16])
        x_recon = self.decoder(Zq)              #torch.Size([128, 3, 64, 64])
        return e_and_q_latent_loss, x_recon, e_latent_loss, q_latent_loss, estimate_codebook_words_exp_entropy
