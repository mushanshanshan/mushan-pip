import torch
import librosa
from librosa.filters import mel as librosa_mel_fn
import numpy as np



hann_window = {}
mel_basis = {}

def spectrogram_torch(y, n_fft=1024, sr=22050, hop_size=256, win_size=1024, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def mel_spectrogram_torch(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spec_to_mel_torch(spec, n_fft, num_mels, sr, fmin=0, fmax=None):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sr,
                             n_fft=n_fft,
                             n_mels=num_mels,
                             fmin=fmin,
                             fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec

def get_data(audio, sr=22050, n_fft=1024, hop_size=256, win_size=1024, n_mel=80):

    if audio.max() > 2:
        audio = torch.FloatTensor(audio.astype(np.float32))
        audio_norm = audio / 2 ** 15
        audio_norm = audio_norm.unsqueeze(0)
    else:
        audio = torch.FloatTensor(audio.astype(np.float32))
        audio_norm = audio / 2 ** 15
        audio_norm = audio_norm.unsqueeze(0)

    print(audio_norm.shape)

    spec = spectrogram_torch(audio_norm,
                             n_fft=n_fft,
                             sr=sr,
                             hop_size=hop_size,
                             win_size=win_size)

    mel = spec_to_mel_torch(spec,
                            n_fft=n_fft,
                            num_mels=n_mel,
                            sr=sr
                            )

    F0 = librosa.pyin(audio_norm.numpy(),
                      fmin=librosa.note_to_hz('C2'),
                      fmax=librosa.note_to_hz('C7'),
                      sr=22050,
                      frame_length=1024,
                      hop_length=256)[0].squeeze(0)



    return audio_norm, spec.squeeze(0), mel.squeeze(0), F0