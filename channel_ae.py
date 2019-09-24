__author__ = 'yihanjiang'
import torch
import torch.nn.functional as F



# entering module for all AWGN based AEs. Input (signal, noise) output (signal_hat, codes)
import commpy.channelcoding.interleavers as RandInterlv
import numpy as np

from numpy.random import mtrand

class Channel_AE(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_AE, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise):
        if self.args.is_interleave == 0:
            pass

        elif self.args.is_same_interleaver == 0:
            interleaver = RandInterlv.RandInterlv(self.args.block_len, np.random.randint(0, 1000))

            p_array = interleaver.p_array
            self.enc.set_interleaver(p_array)
            self.dec.set_interleaver(p_array)

        else:# self.args.is_same_interleaver == 1
            interleaver = RandInterlv.RandInterlv(self.args.block_len, 0) # not random anymore!
            p_array = interleaver.p_array
            self.enc.set_interleaver(p_array)
            self.dec.set_interleaver(p_array)

        codes          = self.enc(input)

        if self.args.channel in ['awgn', 't-dist', 'radar', 'ge_awgn']:
            received_codes = codes + fwd_noise

        elif self.args.channel == 'bec':
            received_codes = codes * fwd_noise

        elif self.args.channel in ['bsc', 'ge']:
            received_codes = codes * (2.0*fwd_noise - 1.0)
            received_codes = received_codes.type(torch.FloatTensor)

        elif self.args.channel == 'fading':
            data_shape = codes.shape
            #  Rayleigh Fading Channel, non-coherent
            fading_h = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) #np.sqrt(2.0)
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
            received_codes = fading_h*codes + fwd_noise

            # fading_h = np.sqrt(np.random.standard_normal(data_shape)**2 +  np.random.standard_normal(data_shape)**2)/np.sqrt(3.14/2.0)
            # noise = sigma * np.random.standard_normal(data_shape) # Define noise
            #
            # # corrupted_signal = 2.0*fading_h*input_signal-1.0 + noise
            # corrupted_signal = fading_h *(2.0*input_signal-1.0) + noise
        else:
            print('default AWGN channel')
            received_codes = codes + fwd_noise

        x_dec          = self.dec(received_codes)

        return x_dec, codes