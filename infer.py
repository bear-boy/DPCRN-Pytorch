import torch
import librosa
import argparse
import utils
import soundfile as sf
import numpy as np
from model import DPCRN

def infer(args, model, hps):
    input, sr = librosa.load(args.audio_path, sr=None)
    l = len(input)
    frames = int(np.ceil((l - hps.train.frame_len) / hps.train.frame_shift)) + 1
    max_t = (frames - 1) * hps.train.frame_shift + hps.train.frame_len
    input_mat = np.zeros((1,max_t), dtype=np.float32)
    input_mat[0, :l] = input
    x = torch.from_numpy(input_mat).type(torch.FloatTensor).cuda()
    y,_,_ = model(x)
    y = y.detach().cpu().data.numpy()
    output = y[:,:l].T
    # out_path = os.path.join(os.curdir, args.save_path)
    sf.write(args.save_path, output, samplerate=sr, subtype='FLOAT')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_path', default='./pretrained_model/final_ckp.pth', type=str, help='Checkpoint path')
    parser.add_argument('--audio_path', default='./dataset/valset_noisy/p232_001.wav', type=str, help='infer audio path')
    parser.add_argument('--save_path', default='enhancement_example/p232_001_enh.wav', type=str, help='save audio path')
    args = parser.parse_args()

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
    net.load_state_dict(torch.load(args.ckp_path))


    infer(args, net, hps)