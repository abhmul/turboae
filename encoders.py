'''
This module contains all possible encoders and utilities
'''

import torch
import torch.nn.functional as F

from numpy import arange
from numpy.random import mtrand
import math
import numpy as np

from interleavers import Interleaver
from utils import snr_db2sigma

# STE implementation
class MyQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args):

        ctx.save_for_backward(inputs)
        ctx.args = args

        if ctx.args.train_channel_mode != 'group_norm_noisy':

            x_lim_abs  = args.enc_value_limit
            x_lim_range = 2.0 * x_lim_abs
            x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

            if args.enc_quantize_level == 2:
                outputs_int = torch.sign(x_input_norm)
            else:
                outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((args.enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(args.enc_quantize_level - 1.0) - x_lim_abs

        else:
            outputs_int = inputs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):

        # STE implementations
        if ctx.args.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>ctx.args.enc_value_limit]=0
            grad_output[input<-ctx.args.enc_value_limit]=0

        if ctx.args.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.args.enc_grad_limit, ctx.args.enc_grad_limit)

        if ctx.args.train_channel_mode not in ['group_norm_noisy', 'group_norm_noisy_quantize']:
            grad_input = grad_output.clone()
        else:
            # PASS GRADIENT NOISE to encoder.
            grad_noise = snr_db2sigma(ctx.args.fb_noise_snr) * torch.randn(grad_output[0].shape, dtype=torch.float)
            ave_temp   = grad_output.mean(dim=0) + grad_noise
            ave_grad   = torch.stack([ave_temp for _ in range(ctx.args.batch_size)], dim=2).permute(2,0,1)
            grad_input = ave_grad + grad_noise

        return grad_input, None

# All encoder's functions.
class ENCBase(torch.nn.Module):
    def __init__(self, args):
        super(ENCBase, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.reset_precomp()

    def set_parallel(self):
        pass

    # precomputing not done yet.
    def set_precomp(self, mean_vec, std_vec):
        self.mean_group = mean_vec.to(self.this_device)
        self.std_group  = std_vec.to(self.this_device)

    # not tested yet
    def reset_precomp(self):
        self.mean_group = torch.zeros(self.args.group_norm_g, self.args.code_rate_n).type(torch.FloatTensor).to(self.this_device)
        self.std_group  = torch.ones(self.args.group_norm_g, self.args.code_rate_n).type(torch.FloatTensor).to(self.this_device)
        self.num_test_block= 0.0

    def enc_act(self, inputs):
        if self.args.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.enc_act == 'elu':
            return F.elu(inputs)
        elif self.args.enc_act == 'relu':
            return F.relu(inputs)
        elif self.args.enc_act == 'selu':
            return F.selu(inputs)
        elif self.args.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.enc_act == 'linear':
            return inputs
        else:
            return inputs

    def power_constraint(self, x_input):

        x_input_shape = x_input.shape
        G             = self.args.group_norm_g

        x_input = x_input.view(int(x_input_shape[0]*x_input_shape[1]/G), G, x_input_shape[2])
        this_mean    = torch.mean(x_input, dim=0)
        this_std     = torch.std(x_input, dim=0)

        x_input = (x_input-this_mean)*1.0 / this_std

        x_input_norm = x_input.view(x_input_shape)

        if self.training:
            # save pretrained mean/std
            try:
                self.num_test_block += 1.0
                self.mean_group = (self.mean_group*(self.num_test_block-1) + this_mean)/self.num_test_block
                self.std_group  = (self.std_group*(self.num_test_block-1) + this_std)/self.num_test_block
            except:
                # with critic length
                print('group normalization seems wired.!')


        if self.args.train_channel_mode == 'group_norm' or self.args.test_channel_mode == 'group_norm':
            pass

        elif self.args.train_channel_mode == 'group_norm_quantize' or self.args.test_channel_mode == 'group_norm_quantize':
            myquantize = MyQuantize.apply
            x_input_norm = myquantize(x_input_norm, self.args)
        else:
            myquantize = MyQuantize.apply
            x_input_norm = myquantize(x_input_norm, self.args)

        return x_input_norm

# Encoder with interleaver. Support different code rate.
class ENC_turbofy_rate2(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_turbofy_rate2, self).__init__(args)
        self.args             = args

        # Encoder
        self.enc_rnn_1       = torch.nn.GRU(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_1    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_rnn_2       = torch.nn.GRU(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_2    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)


    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def forward(self, inputs):
        x_sys, _   = self.enc_rnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_sys_int  = self.interleaver(inputs)

        x_p2, _    = self.enc_rnn_2(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_2(x_p2))

        x_tx       = torch.cat([x_sys, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)
        return codes
######################################################
# Systematic Bit, with rate 1/3, DTA Encocder
######################################################
class ENC_interRNN_sys(ENCBase):
    def __init__(self, args, p_array):
        super(ENC_interRNN_sys, self).__init__(args)
        self.args             = args

        # Encoder
        self.enc_rnn       = torch.nn.GRU(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_rnn_int       = torch.nn.GRU(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_int    = torch.nn.Linear(2*args.enc_num_unit, 1)


        self.interleaver      = Interleaver(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_rnn = torch.nn.DataParallel(self.enc_rnn)
        self.enc_linear = torch.nn.DataParallel(self.enc_linear)

        self.enc_rnn_int = torch.nn.DataParallel(self.enc_rnn_int)
        self.enc_linear_int = torch.nn.DataParallel(self.enc_linear_int)

    def forward(self, inputs):

        x_sys      = 2.0* inputs - 1.0

        x_p1, _    = self.enc_rnn(inputs)
        x_p1       = self.enc_act(self.enc_linear(x_p1))

        x_sys_int  = self.interleaver(inputs)

        x_p2, _    = self.enc_rnn_int(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_int(x_p2))

        x_tx       = torch.cat([x_p1, x_p2], dim = 2)

        x_tx       = self.power_constraint(x_tx)

        codes      = torch.cat([x_sys, x_tx], dim=2)

        return codes


#######################################################
# DTA Encocder, with rate 1/3, RNN only
#######################################################
class ENC_interRNN(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interRNN, self).__init__(args)

        self.enc_rnns    = torch.nn.ModuleList()

        self.args             = args

        # Encoder

        if args.enc_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.enc_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN


        self.enc_rnn_1       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_1    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_rnn_2       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_2    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_rnn_3       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_3    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_rnn_3 = torch.nn.DataParallel(self.enc_rnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):

        x_sys, _   = self.enc_rnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1, _    = self.enc_rnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int  = self.interleaver(inputs)

        x_p2, _    = self.enc_rnn_3(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)
        return codes


#######################################################
# DTA Encocder, with rate 1/3, CNN-1D same shape only
#######################################################
from cnn_utils import SameShapeConv1d
class ENC_interCNN(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interCNN, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_3       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int  = self.interleaver(inputs)

        x_p2       = self.enc_cnn_3(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes

#######################################################
# DTA Encocder, with rate 1/3, CNN-1D same shape only
#######################################################
from cnn_utils import SameShapeConv1d
class ENC_interCNN2Int(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interCNN2Int, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_3       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver1      = Interleaver(args, p_array)


        seed2 = 1000
        rand_gen2 = mtrand.RandomState(seed2)
        p_array2 = rand_gen2.permutation(arange(args.block_len))

        print('p_array1', p_array)
        print('p_array2', p_array2)

        self.interleaver2      = Interleaver(args, p_array2)


    def set_interleaver(self, p_array):
        pass
        #self.interleaver1.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_sys_int1 = self.interleaver1(inputs)

        x_p1       = self.enc_cnn_2(x_sys_int1)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int2 = self.interleaver2(inputs)

        x_p2       = self.enc_cnn_3(x_sys_int2)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes

#######################################################
# DTA Encocder, with rate 1/2, CNN-1D same shape only
#######################################################

class ENC_turbofy_rate2_CNN(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_turbofy_rate2_CNN, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)

    def forward(self, inputs):
        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))


        x_sys_int  = self.interleaver(inputs)

        x_p2       = self.enc_cnn_2(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_2(x_p2))

        x_tx       = torch.cat([x_sys, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes



#######################################################
# DTA Encocder, with rate 1/3, CNN-2D.
#######################################################
from cnn_utils import SameShapeConv2d
from interleavers import  Interleaver2D
class ENC_interCNN2D(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interCNN2D, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv2d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_1    =  torch.nn.Conv2d(args.enc_num_unit, 1, 1, 1, 0, bias=True)

        self.enc_cnn_2       = SameShapeConv2d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_2    = torch.nn.Conv2d(args.enc_num_unit, 1, 1, 1, 0, bias=True)

        self.enc_cnn_3       = SameShapeConv2d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_3    = torch.nn.Conv2d(args.enc_num_unit, 1, 1, 1, 0, bias=True)

        self.interleaver      = Interleaver2D(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        img_size   = int(np.sqrt(self.args.block_len))
        inputs     = inputs.permute(0,2,1).view(self.args.batch_size, 1, img_size, img_size)

        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int  = self.interleaver(inputs)

        x_p2       = self.enc_cnn_3(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 1)
        x_tx = x_tx.view(self.args.batch_size, self.args.code_rate_n, self.args.block_len)
        x_tx = x_tx.permute(0, 2, 1).contiguous()

        codes = self.power_constraint(x_tx)

        return codes


#######################################################
# CNN Encocder, with rate 1/3, CNN-1D same shape only
# No interleaver
#######################################################
class CNN_encoder_rate3(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(CNN_encoder_rate3, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_3       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs):
        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_p2       = self.enc_cnn_3(inputs)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes


#######################################################
# CNN Encocder, with rate 1/2, CNN-1D same shape only
# No interleaver
#######################################################
class CNN_encoder_rate2(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(CNN_encoder_rate2, self).__init__(args)
        self.args             = args

        # Encoder

        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.enc_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)

    def forward(self, inputs):
        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_tx       = torch.cat([x_sys,x_p1], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes






