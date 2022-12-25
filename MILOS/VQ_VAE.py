import torch
import torch.nn as nn
import torch.nn.functional as F
class VectorQuantizer(nn.Module):
    def __init__(self, args_VQ):
        super(VectorQuantizer, self).__init__()
        
        self.D = args_VQ['D']
        self.K = args_VQ['K']
        self.beta = args_VQ['beta']
        
        self.E = nn.Embedding(self.K, self.D)
        self.E.weight.data.uniform_(-1/self.K, 1/self.K)
        
    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # C = D
        Ze_tensor = inputs.permute(0, 2, 3, 1).contiguous()
        Ze_tensor_shape = Ze_tensor.shape
        
        # Flatten input = dims (B*H*W) x D
        Ze = Ze_tensor.view(-1, self.D)
        
        # Calculate distances matrix D = dims (B*H*W) x K
        D = (torch.sum(Ze**2, dim=1, keepdim=True) # sum across axis 1 matrix dims (but keepdims) = [(B*H*W) x D] -> column (B*H*W)-sized vector -> torch broadcast to same K-rows to get matrix = dims (B*H*W) x K
            + torch.sum(self.E.weight**2, dim=1) # sum across axis 1 matrix dims = [K x D] -> row K-sized vector -> torch broadcast to same (B*H*W)-columns to get matrix = dims (B*H*W) x K
            - 2 * torch.matmul(Ze, self.E.weight.t())) # [matrix dims = (B*H*W) x D] * [matrix dims = D x K]^T
            
        # Discretization metric is the closes vector in l2-norm sense
        # Encoding indices = dims (B*H*W) x 1
        encoding_indices = torch.argmin(D, dim=1).unsqueeze(1)
        
        # from embedding matrix E (dims K x D) pick (B*H*W)-number of vectors 
        # put it into original shape of the tensor that VQ got from Encoder, i.e. Ze_tensor_shape
        Zq_tensor = self.E(encoding_indices).view(Ze_tensor_shape)# size = [B, H, W, C]
        
        # Encoder distance tensor to Quantized vectors Loss calculation
        e_latent_loss = F.mse_loss(Zq_tensor.detach(), Ze_tensor) # commitment loss
        
        # Quantized distance tensor to Encoder vectors Loss calculation
        q_latent_loss = F.mse_loss(Zq_tensor,          Ze_tensor.detach())
        
        # Combination of two losses
        e_and_q_latent_loss = q_latent_loss + self.beta * e_latent_loss
        
        
        # in + (out - in).detach() 
        Zq_tensor = Ze_tensor + (Zq_tensor - Ze_tensor).detach()        
        
        # BHWC -> BCHW
        Zq_tensor = Zq_tensor.permute(0, 3, 1, 2).contiguous()
        return e_and_q_latent_loss, Zq_tensor, e_latent_loss.item(), q_latent_loss.item()
    
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
            
        # because blocks finish with one CNN that does not have an activation function put the activation function to be ReLU at the end
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, args_encoder, res_block_args):
        super(Encoder, self).__init__()

        # save Encoder arguments in dict
        self.args_encoder = args_encoder
        
        # Series of conv layers with stride = 2 to reduce the image resolution by a factor of 2 for each conv layer 
        self.sequential_convs = nn.Sequential(
                                                nn.Conv2d(  in_channels=args_encoder['conv1_Cin'],#Cin
                                                            out_channels=args_encoder['conv1_Cout'],#64
                                                            kernel_size=args_encoder['conv1_k'],#4
                                                            stride=args_encoder['conv1_s'],#2
                                                            padding=args_encoder['conv1_p']),#1
                                                nn.ReLU(True),
                                                nn.Conv2d(  in_channels=args_encoder['conv2_Cin'],#64
                                                            out_channels=args_encoder['conv2_Cout'],#128
                                                            kernel_size=args_encoder['conv2_k'],#4
                                                            stride=args_encoder['conv2_s'],#2
                                                            padding=args_encoder['conv2_p']) #, #1
                                                # nn.ReLU(True),
                                                # nn.Conv2d(  in_channels=args_encoder['conv3_Cin'],#128
                                                #             out_channels=args_encoder['conv3_Cout'],#128
                                                #             kernel_size=args_encoder['conv3_k'],#3
                                                #             stride=args_encoder['conv3_s'],#1
                                                #             padding=args_encoder['conv3_p'])#1
                                                #nn.ReLU(True)#input to the residual stack already has ReLU at the input, so we dont put it here (applying ReLU two times does not change anything but increases the computational time)                                            
                                            )
        
        # Series of two-depth residual blocks packed into a residual stack (e.g. ResNet)
        self.residual_stack = ResidualStack(res_block_args)                                             
        
        #
        self.channel_adjusting_conv = nn.Conv2d(in_channels=args_encoder['channel_adjusting_conv_Cin'], #C_enc, 
                                                out_channels=args_encoder['channel_adjusting_conv_Cout'], #D,
                                                kernel_size=args_encoder['channel_adjusting_conv_k'],#1, # kernel size has to be 1 because this (linear) conv layer just changes the chanell dimension
                                                stride=args_encoder['channel_adjusting_conv_s'],#1,
                                                padding=args_encoder['channel_adjusting_conv_p']#0
                                                )
        if args_encoder['M'] == 0:
            # First we calculate the padding
            padding = (0,1,0,1) #['W_left'], ['Padding_W_right'], ['Padding_H_top'], ['Padding_H_bottom']
            self.zeropad1=  nn.ZeroPad2d(padding)

    def forward(self, x):
        x = self.sequential_convs(x)
        x = self.residual_stack(x)
        x = self.channel_adjusting_conv(x)
        if self.args_encoder['M'] == 0:
            x = self.zeropad1(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, args_decoder, res_block_args):
        super(Decoder, self).__init__()
        
        self.channel_adjusting_conv = nn.Conv2d(in_channels=args_decoder['channel_adjusting_conv_Cin'],#D
                                                out_channels=args_decoder['channel_adjusting_conv_Cout'],#128
                                                kernel_size=args_decoder['channel_adjusting_conv_k'],#3 ; TRY WITH KERNEL SIZE 1 AND PADDING 0
                                                stride=args_decoder['channel_adjusting_conv_s'],#1
                                                padding=args_decoder['channel_adjusting_conv_p']#1 ; TRY WITH KERNEL SIZE 1 AND PADDING 0
                                                )
        
        self.residual_stack = ResidualStack(res_block_args)
        
        self.sequential_trans_convs = torch.nn.Sequential()
        
        for i in range(1,args_decoder['CONV_LAYERS']+1):
            self.sequential_trans_convs.add_module(f"conv2d_{i}", nn.ConvTranspose2d(   in_channels=args_decoder[f'trans_conv{i}_Cin'],#64
                                                                                        out_channels=args_decoder[f'trans_conv{i}_Cout'],#3
                                                                                        kernel_size=args_decoder[f'trans_conv{i}_k'],#4
                                                                                        stride=args_decoder[f'trans_conv{i}_s'],#2
                                                                                        padding=args_decoder[f'trans_conv{i}_p'])#1
                                                   )
            
            if i != args_decoder['CONV_LAYERS']:
                # Do not add ReLU() activation function layer to the last CNN
                self.sequential_trans_convs.add_module(f"ReLU_{i}", nn.ReLU(True))
        
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
        x = self.channel_adjusting_conv(x)
        x = self.residual_stack(x)
        x = self.sequential_trans_convs(x)    
        return x

class VQ_VAE(nn.Module):
    def __init__(self, args_encoder, args_VQ, args_decoder, res_block_args):
        super(VQ_VAE, self).__init__()
        
        ######################
        # Model Constructors #
        ######################
        self.C_in, self.H_in, self.W_in = args_encoder['C_in'], args_encoder['H_in'], args_encoder['W_in']
        self.args_encoder=args_encoder
        self.args_VQ=args_VQ
        self.args_decoder=args_decoder
        self.res_block_args = res_block_args
        
        self.encoder =  Encoder(args_encoder, res_block_args)
        self.VQ      =  VectorQuantizer(args_VQ)
        self.decoder =  Decoder(args_decoder, res_block_args)

    def forward(self, x):                       #torch.Size([128, 3, 64, 64])
        Ze = self.encoder(x)                    #torch.Size([128, 64, 16, 16])
        e_and_q_latent_loss, Zq, e_latent_loss, q_latent_loss = self.VQ(Ze)   #torch.Size([128, 64, 16, 16])
        # use this for simple countinous vanilla AE (i.e. to bypass the VQ class forward pass)
        #e_and_q_latent_loss, Zq,  e_latent_loss, q_latent_loss = 0,Ze, 0, 0         #torch.Size([128, 64, 16, 16])
        x_recon = self.decoder(Zq)              #torch.Size([128, 3, 64, 64])
        return e_and_q_latent_loss, x_recon, e_latent_loss, q_latent_loss