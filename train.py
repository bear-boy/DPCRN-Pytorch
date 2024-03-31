'''
    training script for DPCRN
'''
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import DPCRN, STFT
import utils
from se_dataset import AudioDataset
from loss import DPCRNLoss
from EarlyStopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='./logs/DPCRN/', type=str, help='Log file path to record training status')
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
args = parser.parse_args()

def get_lr(optim):
    return optim.param_groups[0]['lr']

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

    stft = STFT(hps.train.frame_len, hps.train.frame_shift)

    # checkpoints load
    if (os.path.exists('./final.pth.tar')):
        checkpoint = torch.load('./final.pth.tar')
        net.load_state_dict(checkpoint)

    # log
    writer = SummaryWriter(args.log_dir)

    train_dataset = AudioDataset(data_type='train',win_len=hps.train.frame_len,hop_len=hps.train.frame_shift)
    val_dataset = AudioDataset(data_type='val',win_len=hps.train.frame_len,hop_len=hps.train.frame_shift)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
            collate_fn=train_dataset.collate, shuffle=True, num_workers=1)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
            collate_fn=val_dataset.collate, shuffle=False, num_workers=1)

    torch.set_printoptions(precision=10, profile="full")

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=10e-10, cooldown=1)
    # Early-Stopping
    early_stopping = EarlyStopping(patience=10,verbose=True,path='./logs/DPCRN/ckp/')

    writer_step = 0
    for epoch in range(args.num_epochs):
        # training
        net.train()
        train_bar = tqdm(train_data_loader)
        step = 0
        record_loss = 0.0
        for input in train_bar:
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
            # train_mixed, train_clean, seq_len = input
            train_clean_re, train_clean_im = stft(train_clean)
            en_sp, en_re, en_im = net(train_mixed)
            for i, l in enumerate(seq_len):
                en_sp[i, l:] = 0
            loss = DPCRNLoss(train_clean, en_sp, train_clean_re, train_clean_im, en_re, en_im)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            record_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print('Step {} in Epoch [{}/{}], training_loss: {:.4f}'.format(step, epoch, args.num_epochs, loss.item()))
                writer.add_scalar('training loss', loss.item(), writer_step)
                writer_step += 1

        train_loss = record_loss / len(train_data_loader)
        print('#Epoch [{}/{}], training_loss: {:.4f}'.format(epoch, args.num_epochs, train_loss))
        writer.add_scalar('lr', get_lr(optimizer), epoch)
        writer.add_scalar('train_loss', train_loss, epoch)

        # validation
        with torch.no_grad():
            net.eval()
            val_bar = tqdm(val_data_loader)
            val_loss = 0.0
            for input in val_bar:
                val_mixed, val_clean, seq_len = map(lambda x: x.cuda(), input)
                val_clean_re, val_clean_im = stft(val_clean)
                en_sp, en_re, en_im = net(val_mixed)

                loss = DPCRNLoss(val_clean, en_sp, val_clean_re, val_clean_im, en_re, en_im)
                val_loss += loss.item()

            val_loss = val_loss / len(val_data_loader)
            print('Epoch [{}/{}], validation_loss: {:.4f}'.format(epoch, args.num_epochs, val_loss))
            writer.add_scalar('val_loss', val_loss, epoch)

        # learning-rate scheduler
        scheduler.step(val_loss)
        # Early-Stopping Check
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    # torch.save(net.state_dict(), './logs/DPCRN/final.pth')

if __name__ == '__main__':
    main()