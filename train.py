'''
    training script for DPCRN
'''

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from model import DPCRN, STFT
import utils
from se_dataset import AudioDataset
from loss import DPCRNLoss
from conv_stft import ConvSTFT

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
args = parser.parse_args()


def main():
    hps = utils.get_hparams()
    net = DPCRN(encoder_in_channel=hps.train.encoder_in_channel,
                  encoder_channel_size=hps.train.encoder_channel_size,
                  encoder_kernel_size=hps.train.encoder_kernel_size,
                  encoder_stride_size=hps.train.encoder_stride_size,
                  encoder_padding=hps.train.encoder_padding,
                  decoder_in_channel=hps.train.decoder_in_channel,
                  decoder_channel_size=hps.train.decoder_channel_size,
                  decoder_kernel_size=hps.train.decoder_kernel_size,
                  decoder_stride_size=hps.train.decoder_stride_size,
                  rnn_type=hps.train.dprnn_rnn_type,
                  input_size=hps.train.dprnn_input_size,
                  hidden_size=hps.train.dprnn_hidden_size,
                  frame_len=hps.train.frame_len,
                  frame_shift=hps.train.frame_shift).cuda()

    stft = STFT(hps.train.frame_len, hps.train.frame_shift).cuda()

    # checkpoints load
    if (os.path.exists('./final.pth.tar')):
        checkpoint = torch.load('./final.pth.tar')
        net.load_state_dict(checkpoint)

    train_dataset = AudioDataset(data_type='train')
    val_dataset = AudioDataset(data_type='val')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
            collate_fn=train_dataset.collate, shuffle=True, num_workers=1)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
            collate_fn=val_dataset.collate, shuffle=False, num_workers=1)

    torch.set_printoptions(precision=10, profile="full")

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # Learning rate scheduler
    scheduler = ExponentialLR(optimizer, 0.95)

    for epoch in range(args.num_epochs):
        train_bar = tqdm(train_data_loader)
        record_loss = 0.0
        for input in train_bar:
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
            # train_mixed, train_clean, seq_len = input
            train_clean_re, train_clean_im = stft(train_clean)
            en_sp, en_re, en_im = net(train_mixed)

            loss = DPCRNLoss(train_clean, en_sp, train_clean_re, train_clean_im, en_re, en_im)
            # print(epoch, loss)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            record_loss = loss.detach()

        print(epoch, record_loss)
        scheduler.step()
    torch.save(net.state_dict(), './final.pth.tar')

if __name__ == '__main__':
    main()
