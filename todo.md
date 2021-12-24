1. Train from scratch new TurboAE-cont

### Get SNR=1dB good continuous code: TurboAE-continuous.
- in howtos.md he trains with -enc_num_layer 5, but paper uses 2. Let's use 2 to reproduce his results in paper.
Start from scratch by running: 

    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 2 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 800 --print_test_traj -loss bce

This will take 1-1.5 days on nVidia 1080 Ti to train. Take some rest, hiking or watch a season of Anime during waiting.
After converged, it will show the model is saved in `./tmp/*.pt`.
Then incrase batch size by a factor of 2, and train for another 100 epochs:

    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 100 --print_test_traj -loss bce -init_nw_weight ./tmp/*.pt

When saturates, increase the batch size, till you hit some memory limit. Then reduce the learning rate (mult by 0.1) for both encoder and decoder.

2. Fine-tun turboae cont

### Fine-tune at other SNRs
Change encoder training SNR to testing SNR, for example, change to `-train_enc_channel_low 2.0 -train_enc_channel_high 2.0 ` will train good code at 2dB. 
I find the training SNR for decoder fixed to `-1.5 to 2dB` seems works best.

3. Test Turboae-cont
- according to paper, --precompute_norm_stats should be set


4. Binarize and finetune

### Binarize Code via STE, to generate TurboAE-binary.
Use a well-trained continuous code as initialization, and finetune with STE as:

    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm_ste -test_channel_mode block_norm_ste -num_epoch 200 --print_test_traj -loss bce -init_nw_weight ./tmp/*.pt

Also use increasing batch size, and reducing learning rate till converges. Train model for each SNR. 
Typically after 200-400 epochs, STE-trained model converges. Note: this actually replicates **Figure 4 right**.

4. Test models

