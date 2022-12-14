from ..audio.spec_process import get_data
import librosa
import numpy as np

def plot_spectrogram(audio, pitc_scale=1, pitch_mel=True, durations=None, level='f'):
    import matplotlib.pylab as plt
    
    naudio, spec, melspec, F0 = get_data(audio)
    
    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(melspec, aspect="auto", origin="lower",
                   interpolation='none')

    freqs = librosa.core.mel_frequencies(fmin=0.0, fmax=8000, n_mels=80)
    if pitch_mel:
        F0_mel = [np.argmin(abs(freqs - p))*pitc_scale for p in F0]
    else:
        F0_mel = F0 * pitc_scale
    
    plt.plot([i for i in range(len(F0_mel))], F0_mel, color='r', linewidth=2.5)

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.show()