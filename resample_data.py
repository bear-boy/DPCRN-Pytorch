import librosa
import os
import soundfile as sf
from tqdm import tqdm

clean_path = './testset_clean_to_be_resampled/'
noisy_path = './testset_noisy_to_be_resampled/'

clean_samples = os.listdir(clean_path)
noisy_samples = os.listdir(noisy_path)

clean_samples = [f for f in clean_samples if f.endswith('.wav')]
noisy_samples = [f for f in noisy_samples if f.endswith('.wav')]

output_clean_path = './dataset/valset_clean/'
output_noisy_path = './dataset/valset_noisy/'

for id in tqdm(range(len(clean_samples))):
    # p = clean_path + clean_samples[id]
    y, sr = librosa.load(clean_path + clean_samples[id], sr=None)
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sf.write(output_clean_path + clean_samples[id], y_16k, samplerate=16000)

for id in tqdm(range(len(noisy_samples))):
    # p = noisy_path + noisy_samples[id]
    y, sr = librosa.load(noisy_path + noisy_samples[id], sr=None)
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sf.write(output_noisy_path + noisy_samples[id], y_16k, samplerate=16000)