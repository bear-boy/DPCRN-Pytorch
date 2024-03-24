import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DPRNN
from conv_stft import ConvSTFT, ConviSTFT
import utils
import numpy as np

class STFT(nn.Module):
    def __init__(self, frame_len, frame_hop, fft_len=None):
        super(STFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(frame_len)))

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(x, n_fft=self.fft_len, hop_length=self.frame_hop,
                       win_length=self.frame_len, return_complex=True)
        r = y.real
        i = y.imag
        return r,i

class ISTFT(nn.Module):
    def __init__(self, frame_len, frame_hop, fft_len=None):
        super(ISTFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(frame_len)))

    def forward(self, real, imag):
        x = torch.complex(real, imag)
        y = torch.istft(x, n_fft=self.fft_len, hop_length=self.frame_hop,
                        win_length=self.frame_len)
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
        inputs = torch.stack([re,im],dim=1).permute(0,1,3,2)     # B x C x T x F
        x, skips = self.encoder(inputs)

        x = x.permute(0,1,3,2)                  # B x C x F x T
        x = self.dprnn(x)
        x = x.permute(0,1,3,2)                  # B x C x T x F

        mask = self.decoder(x, skips)
        en_re, en_im = self.mask_speech(mask, inputs)      # en_cplx shape: B x T x F
        en_speech = self.istft(en_re.permute(0,2,1), en_im.permute(0,2,1))
        return en_speech, en_re, en_im

    def mask_speech(self, mask, x):
        mask_re = mask[:,0,:,:]
        mask_im = mask[:,1,:,:]

        x_re = x[:,0,:,:]
        x_im = x[:,0,:,:]

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
        # x shape: B x C x T x F
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
                                                    padding=[0, 1], output_padding=[0, 1]))
            else:
                self.conv.append(nn.ConvTranspose2d(in_channels=in_chan, out_channels=channel_size[i],
                                                    kernel_size=kernel_size[i], stride=stride_size[i],
                                                    padding=[0,1]))
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i] * 2
        self.prelu = nn.PReLU()

    def forward(self, x, skips):
        # x shape: B x C x T x F
        for i, (layer, norm, skip) in enumerate(zip(self.conv, self.norm, reversed(skips))):
            x = torch.cat([x,skip], dim=1)
            x = layer(x)[:,:,:-1,:]
            x = self.prelu(norm(x))
        return x

def test():
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
    model.eval()
    batch_size = 16
    x = torch.randn((batch_size, 16000*5)) # 5s inputs
    y = model(x)
    return y

if __name__ == "__main__":
    test()
