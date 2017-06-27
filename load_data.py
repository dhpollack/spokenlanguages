import numpy as np
import librosa

def load_csv(filelist = "data/trainingset.csv"):
    l = []
    with open(filelist, "r") as f:
        for line in f.readlines():
            fp, lab = line.strip().split(",")
            l.append([fp, lab])
    return(l)
def process_audio_files(l, base_dir = "data/train/", N=100, lfilter = None):
    sigs = []
    srs = []
    labels = []
    for fname, label in l[:N]:
        if (lfilter == None) or (label in lfilter):
            filename = base_dir + fname
            sig, sr = librosa.core.load(filename, sr=None)
            sig = np.trim_zeros(sig)
            sigs.append(sig)
            srs.append(sr)
            labels.append(label)
    return(sigs, srs, labels)

def get_mel_spectrograms(sigs, srs, wsize = 2**10, verbose = True):
    hsize = wsize//2
    spectrograms = []
    shapes = set()
    for sig, sr in zip(sigs, srs):
        S_mel = librosa.feature.melspectrogram(sig, sr=sr, n_fft=wsize, hop_length=hsize, n_mels=128)
        shapes.add(S_mel.shape)
        spectrograms.append(S_mel)
    min_shape = min(shapes)
    spectrograms = np.array([x[:,:min_shape[1]] for x in spectrograms])
    if verbose:
        win_times = np.unique(1000 * wsize / np.array(srs))
        hop_times = np.unique(1000 * hsize / np.array(srs))
        print('Window Length: %.2fms'%win_times)
        print('Hop Length: %.2fms'%hop_times)
    return(spectrograms)

def get_chromagrams(sigs, srs, wsize = 2**10):
    chromagrams = []
    shapes = set()
    for sig, sr in zip(sigs, srs):
        chroma = librosa.feature.chroma_stft(sig, sr=sr, n_fft=wsize, hop_length=wsize//2, n_chroma=128)
        shapes.add(chroma.shape)
        chromagrams.append(chroma)
    min_shape = min(shapes)
    chromagrams = np.array([x[:,:min_shape[1]] for x in chromagrams])
    return(chromagrams)

#audio_file_names = load_csv()
#spectrograms, sample_rates, labels = get_mel_spectrograms(audio_file_names)
#print(len(audio_file_names))
#print(spectrograms.shape)
