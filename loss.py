'''
    Loss function for DPCRN
'''

import torch
import torch.nn as nn

def spec_loss(y_true_re, y_true_im, y_pred_re, y_pred_im):
    mse_fn = nn.MSELoss()
    # real-part loss
    real_loss = mse_fn(y_true_re, y_pred_re)
    # imag-part loss
    imag_loss = mse_fn(y_true_im, y_pred_im)
    # magnitude loss
    y_true_mag = (y_true_re ** 2 + y_true_im ** 2 + 1e-8) ** 0.5
    y_pred_mag = (y_pred_re ** 2 + y_pred_im ** 2 + 1e-8) ** 0.5
    mag_loss = mse_fn(y_true_mag, y_pred_mag)

    total_loss = real_loss + imag_loss + mag_loss
    total_loss = torch.log(total_loss + 1e-8)
    return total_loss

def snr_loss(y_true, y_pred):
    snr = torch.mean(torch.square(y_true),dim=-1,keepdim=True) / (torch.mean(torch.square(y_pred-y_true),dim=-1,keepdim=True) +1e-8)
    loss = -10 * torch.log10(snr + 1e-8)
    loss = torch.squeeze(loss,-1).mean()
    return loss


def DPCRNLoss(y_true, y_pred, y_true_re, y_true_im, y_pred_re, y_pred_im):
    loss1 = snr_loss(y_true, y_pred)
    loss2 = spec_loss(y_true_re, y_true_im, y_pred_re, y_pred_im)
    loss = loss1 + loss2
    return loss


if __name__ == "__main__":
    from conv_stft import ConvSTFT
    stft = ConvSTFT(win_len=400,win_inc=100)

    batch_size = 16
    y_true = torch.randn((batch_size, 16000 * 2))  # 2s inputs
    y_true_re ,y_true_im = stft(y_true)

    y_pred = torch.randn((batch_size, 16000 * 2))
    y_pred_re, y_pred_im = stft(y_pred)

    loss = DPCRNLoss(y_true, y_pred, y_true_re, y_true_im, y_pred_re, y_pred_im)
    print(loss)