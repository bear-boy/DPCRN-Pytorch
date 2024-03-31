import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DPRNN
from conv_stft import ConvSTFT, ConviSTFT
import utils
import scipy.signal as signal
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cos_win = torch.from_numpy(signal.windows.cosine(400,False)).type(torch.FloatTensor).cuda()


class STFT(nn.Module):
    def __init__(self, frame_len, frame_hop, fft_len=None):
        super(STFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(x, hop_length=self.frame_hop,
                       n_fft=self.frame_len, window=cos_win, return_complex=True, center=False)
        r = y.real
        i = y.imag
        return r,i

class ISTFT(nn.Module):
    def __init__(self, frame_len, frame_hop, fft_len=None):
        super(ISTFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, real, imag):
        x = torch.complex(real, imag)
        y = torch.istft(x, hop_length=self.frame_hop,
                        n_fft=self.frame_len, window=cos_win,center=False)
        return y

class DPCRN(nn.Module):
    def __init__(self, encoder_in_channel, encoder_channel_size, encoder_kernel_size, encoder_stride_size, encoder_padding,
                       decoder_in_channel, decoder_channel_size, decoder_kernel_size, decoder_stride_size,
                       rnn_type, input_size, hidden_size,
                       frame_len, frame_shift):
        super(DPCRN, self).__init__()
        self.encoder_channel_size = encoder_channel_size
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_stride_size = encoder_stride_size
        self.encoder_padding = encoder_padding
        self.decoder_channel_size = decoder_channel_size
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
        self.frame_len = frame_len
        self.frame_shift = frame_shift

        # self.stft = ConvSTFT(win_len=frame_len, win_inc=frame_shift)
        # self.istft = ConviSTFT(win_len=frame_len, win_inc=frame_shift)
        self.stft = STFT(self.frame_len, self.frame_shift)
        self.istft = ISTFT(self.frame_len, self.frame_shift)

        self.encoder = Encoder(encoder_in_channel, self.encoder_channel_size,
                               self.encoder_kernel_size, self.encoder_stride_size, self.encoder_padding)
        self.decoder = Decoder(decoder_in_channel, self.decoder_channel_size,
                               self.decoder_kernel_size, self.decoder_stride_size)
        self.dprnn = DPRNN(rnn_type, input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        re, im = self.stft(x)
        inputs = torch.stack([re,im],dim=1)     # B x C x F x T
        x, skips = self.encoder(inputs)

        x = self.dprnn(x)

        mask = self.decoder(x, skips)
        en_re, en_im = self.mask_speech(mask, inputs)      # en_ shape: B x F x T
        en_speech = self.istft(en_re, en_im)
        return en_speech, en_re, en_im

    def mask_speech(self, mask, x):
        mask_re = mask[:,0,:,:]
        mask_im = mask[:,1,:,:]

        x_re = x[:,0,:,:]
        x_im = x[:,1,:,:]

        en_re = x_re * mask_re - x_im * mask_im
        en_im = x_re * mask_im + x_im * mask_re
        return en_re, en_im

class Encoder(nn.Module):
    def __init__(self, in_channel_size, channel_size, kernel_size, stride_size, padding):
        super(Encoder, self).__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding = padding

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        in_chan = in_channel_size
        for i in range(len(channel_size)):
            self.conv.append(nn.Conv2d(in_channels=in_chan,out_channels=channel_size[i],
                                       kernel_size=kernel_size[i], stride=stride_size[i]))
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i]
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x shape: B x C x F x T
        skips = []
        for i, (layer, norm) in enumerate(zip(self.conv, self.norm)):
            x = F.pad(x, pad=self.padding[i])
            x = layer(x)
            x = self.prelu(norm(x))
            skips.append(x)
        return x, skips

class Decoder(nn.Module):
    def __init__(self, in_channel_size, channel_size, kernel_size, stride_size):
        super(Decoder, self).__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        in_chan = in_channel_size
        for i in range(len(channel_size)):
            if i == 3:
                self.conv.append(nn.ConvTranspose2d(in_channels=in_chan, out_channels=channel_size[i],
                                                    kernel_size=kernel_size[i], stride=stride_size[i],
                                                    padding=[1, 0], output_padding=[1, 0]))
            else:
                self.conv.append(nn.ConvTranspose2d(in_channels=in_chan, out_channels=channel_size[i],
                                                    kernel_size=kernel_size[i], stride=stride_size[i],
                                                    padding=[1,0]))
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i] * 2
        self.prelu = nn.PReLU()

    def forward(self, x, skips):
        # x shape: B x C x F x T
        for i, (layer, norm, skip) in enumerate(zip(self.conv, self.norm, reversed(skips))):
            x = torch.cat([x,skip], dim=1)
            x = layer(x)[:,:,:,:-1]
            x = self.prelu(norm(x))
        return x

def test_model():
    hps = utils.get_hparams()
    model = DPCRN(encoder_in_channel=hps.train.encoder_in_channel,
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
                  frame_shift=hps.train.frame_shift)
    model = model.to(device)
    model.eval()
    batch_size = 16
    x = torch.randn((batch_size, 16000*5)).to(device) # 5s inputs
    y = model(x)
    return y

def test_stft():
    hps = utils.get_hparams()
    stft = STFT(frame_len=hps.train.frame_len, frame_hop=hps.train.frame_shift)
    istft = ISTFT(frame_len=hps.train.frame_len, frame_hop=hps.train.frame_shift)
    x = torch.randn((8,16100*5))
    x_r, x_i = stft(x)
    x_rec = istft(x_r, x_i)
    print(x_rec.size(1))

def get_model_size():
    hps = utils.get_hparams()
    model = DPCRN(encoder_in_channel=hps.train.encoder_in_channel,
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
                  frame_shift=hps.train.frame_shift)
    model = model.to(device)
    para = [p.numel() for p in model.parameters() if p.requires_grad]
    total_para_size = sum(para)
    print(total_para_size)
    return total_para_size

if __name__ == "__main__":
    # get_model_size()
    # test_stft()
    test_model()