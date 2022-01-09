__author__ = 'yihanjiang'

import torch
from utils import snr_db2sigma, snr_sigma2db
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class NonIIDMarkovianGaussianAsAWGN:
    def __init__(self, sigma, block_len, p_gb=0.8, p_bg=0.8):
        self.p_gb = p_gb
        self.p_bg = p_bg
        print("self.p_gb: " + str(self.p_gb))
        print("self.p_bg: " + str(self.p_bg))
        self.block_len = block_len
        self.sigma = sigma

        self.snr = snr_sigma2db(self.sigma)
        self.good_snr = self.snr + 1
        self.bad_snr = self.snr - 1
        print("self.good_snr: " + str(self.good_snr))
        print("self.bad_snr: " + str(self.bad_snr))
        # self.good_snr = self.snr
        # self.bad_snr = self.snr
        self.good_sigma = snr_db2sigma(self.good_snr)
        self.bad_sigma = snr_db2sigma(self.bad_snr)
        print("self.good_sigma: " + str(self.good_sigma))
        print("self.bad_sigma: " + str(self.bad_sigma))
        print((self.sigma, self.good_sigma, self.bad_sigma))
        # Always start in good state; good = 1, bad = 0
        self.initial_distribution = tfd.Categorical(probs=[0.0, 1.0])
        self.transition_distribution = tfd.Categorical(probs=[[1 - p_bg, p_bg],
                                                              [p_gb, 1 - p_gb]])
        self.observation_distribution = tfd.Normal(loc=[0., 0.], scale=[self.bad_sigma, self.good_sigma])
        self.distribution = tfd.HiddenMarkovModel(self.initial_distribution, self.transition_distribution, self.observation_distribution, num_steps=self.block_len)
    
    def noise_func(self, shape):
        # shape[1] corresponds to time. we sample Batch x Channels x Time then swap channels and time axes
        return tf.transpose(self.distribution.sample((shape[0], shape[2])), perm=[0, 2, 1]).numpy()

class AdditiveTonAWGN:
    def __init__(self, sigma, v=3):
        self.sigma = sigma
        self.v = v
        self.distribution = tfp.distributions.StudentT(df=self.v, loc=0, scale=1)
        print('self.v: ' + str(self.v))
        print('self.sigma: ' + str(self.sigma))
    
    def noise_func(self, shape):
        return self.sigma * tf.sqrt((self.v - 2) / self.v) * self.distribution.sample(shape)

# test_sigma is really test_snr
def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0, mode = 'encoder'):
    # SNRs at training
    if test_sigma == 'default':  # This is the case that is run
        if args.channel == 'bec':
            if mode == 'encoder':
                this_sigma = args.bec_p_enc
            else:
                this_sigma = args.bec_p_dec

        elif args.channel in ['bsc', 'ge']:
            if mode == 'encoder':
                this_sigma = args.bsc_p_enc
            else:
                this_sigma = args.bsc_p_dec
        else: # general AWGN cases  - For channel = awgn we go here
            this_sigma_low = snr_db2sigma(snr_low)
            this_sigma_high= snr_db2sigma(snr_high)
            # mixture of noise sigma.
            this_sigma = (this_sigma_low - this_sigma_high) * torch.rand(noise_shape) + this_sigma_high

    else:
        if args.channel in ['bec', 'bsc', 'ge']:  # bsc/bec noises
            this_sigma = test_sigma
        else:
            this_sigma = snr_db2sigma(test_sigma)

    print(test_sigma)
    # SNRs at testing
    if args.channel == 'awgn':  # This is our case
        print("generating noise for awgn")
        # assert snr_low == snr_high
        print("snr: " + str(test_sigma))
        noniid_awgn_sigma = snr_db2sigma(test_sigma)
        print("noniid_awgn_sigma: " + str(noniid_awgn_sigma))
        print("awgn_sigma: " + str(this_sigma))
        assert isinstance(noise_shape, tuple)
        print("shape " + str(noise_shape))
        print("block_len: " + str(args.block_len))
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 't-dist':
        print("generating noise for atn")
        print("atn_sigma: " + str(this_sigma))
        print("atn_snr: " + str(snr_sigma2db(this_sigma)))
        print("shape " + str(noise_shape))
        # fwd_noise  = this_sigma * torch.from_numpy(np.sqrt((args.vv-2)/args.vv) * np.random.standard_t(args.vv, size = noise_shape)).type(torch.FloatTensor)
        my_channel = AdditiveTonAWGN(this_sigma, v=args.vv)
        fwd_noise = torch.from_numpy(my_channel.noise_func(noise_shape).numpy()).type(torch.FloatTensor)

    elif args.channel == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], noise_shape,
                                       p=[1 - args.radar_prob, args.radar_prob])

        corrupted_signal = args.radar_power* np.random.standard_normal( size = noise_shape ) * add_pos
        fwd_noise = this_sigma * torch.randn(noise_shape, dtype=torch.float) +\
                    torch.from_numpy(corrupted_signal).type(torch.FloatTensor)

    elif args.channel == 'bec':
        fwd_noise = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape,
                                        p=[this_sigma, 1 - this_sigma])).type(torch.FloatTensor)

    elif args.channel == 'bsc':
        fwd_noise = torch.from_numpy(np.random.choice([0.0, 1.0], noise_shape,
                                        p=[this_sigma, 1 - this_sigma])).type(torch.FloatTensor)
    elif args.channel == 'ge_awgn':
        #G-E AWGN channel
        # p_gg = 0.8         # stay in good state
        # p_bb = 0.8
        # bsc_k = snr_db2sigma(snr_sigma2db(this_sigma) + 1)          # accuracy on good state
        # bsc_h = snr_db2sigma(snr_sigma2db(this_sigma) - 1)   # accuracy on good state

        # fwd_noise = np.zeros(noise_shape)
        # for batch_idx in range(noise_shape[0]):
        #     for code_idx in range(noise_shape[2]):

        #         good = True
        #         for time_idx in range(noise_shape[1]):
        #             if good:
        #                 if test_sigma == 'default':
        #                     fwd_noise[batch_idx,time_idx, code_idx] = bsc_k[batch_idx,time_idx, code_idx]
        #                 else:
        #                     fwd_noise[batch_idx,time_idx, code_idx] = bsc_k
        #                 good = np.random.random()<p_gg
        #             elif not good:
        #                 if test_sigma == 'default':
        #                     fwd_noise[batch_idx,time_idx, code_idx] = bsc_h[batch_idx,time_idx, code_idx]
        #                 else:
        #                     fwd_noise[batch_idx,time_idx, code_idx] = bsc_h
        #                 good = np.random.random()<p_bb
        #             else:
        #                 print('bad!!! something happens')

        # fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)* torch.randn(noise_shape, dtype=torch.float)
        print("generating noise for awgn")
        # assert snr_low == snr_high
        print("snr: " + str(test_sigma))
        noniid_awgn_sigma = snr_db2sigma(test_sigma)
        print("noniid_awgn_sigma: " + str(noniid_awgn_sigma))
        print("awgn_sigma: " + str(this_sigma))
        assert isinstance(noise_shape, tuple)
        print("shape " + str(noise_shape))
        print("block_len: " + str(args.block_len))
        my_channel = NonIIDMarkovianGaussianAsAWGN(noniid_awgn_sigma, args.block_len)
        fwd_noise = torch.from_numpy(my_channel.noise_func(noise_shape)).type(torch.FloatTensor)

    elif args.channel == 'ge':
        #G-E discrete channel
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = 1.0        # accuracy on good state
        bsc_h = this_sigma# accuracy on good state

        fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        tmp = np.random.choice([0.0, 1.0], p=[1-bsc_k, bsc_k])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_gg
                    elif not good:
                        tmp = np.random.choice([0.0, 1.0], p=[ 1-bsc_h, bsc_h])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)

    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise



