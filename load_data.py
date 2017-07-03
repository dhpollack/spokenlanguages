import numpy as np
import librosa

def load_csv(filelist = "data/trainingset.csv"):
    l = []
    with open(filelist, "r") as f:
        for line in f.readlines():
            fp, lab = line.strip().split(",")
            l.append([fp, lab])
    return(l)
def process_audio_files(l, base_dir = "data/train/", N=None, lfilter = None):
    sigs = []
    srs = []
    labels = []
    l = l[:N] if l is not None else l
    for fname, label in l:
        if (lfilter == None) or (label in lfilter):
            filename = base_dir + fname
            sig, sr = librosa.core.load(filename, sr=None)
            sig = np.trim_zeros(sig)
            sigs.append(sig)
            srs.append(sr)
            labels.append(label)
    return(sigs, srs, labels)

def get_mel_spectrograms(sigs, srs, wsize = 2**10, freq_bands = 128, log10 = False, verbose = True):
    hsize = wsize//2
    spectrograms = []
    shapes = set()
    for sig, sr in zip(sigs, srs):
        S_mel = librosa.feature.melspectrogram(sig, sr=sr, n_fft=wsize, hop_length=hsize, n_mels=freq_bands)
        shapes.add(S_mel.shape)
        spectrograms.append(S_mel)
    min_shape = min(shapes)
    spectrograms = np.array([x[:,:min_shape[1]] for x in spectrograms])
    if log10:
        spectrograms = np.log10(spectrograms)
    if verbose:
        win_times = np.unique(1000 * wsize / np.array(srs))
        hop_times = np.unique(1000 * hsize / np.array(srs))
        print('Window Length: %.2fms'%win_times)
        print('Hop Length: %.2fms'%hop_times)
    return(spectrograms)

def get_chromagrams(sigs, srs, wsize = 2**10, freq_bands = 128, log10 = False, verbose = True):
    hsize = wsize // 2
    chromagrams = []
    shapes = set()
    for sig, sr in zip(sigs, srs):
        chroma = librosa.feature.chroma_stft(sig, sr=sr, n_fft=wsize, hop_length=hsize, n_chroma=freq_bands)
        shapes.add(chroma.shape)
        chromagrams.append(chroma)
    min_shape = min(shapes)
    chromagrams = np.array([x[:,:min_shape[1]] for x in chromagrams])
    if log10:
        chromagrams = np.log10(chromagrams)
    if verbose:
        win_times = np.unique(1000 * wsize / np.array(srs))
        hop_times = np.unique(1000 * hsize / np.array(srs))
        print('Window Length: %.2fms'%win_times)
        print('Hop Length: %.2fms'%hop_times)
    return(chromagrams)

def get_grams(use_chromagrams = False, load_grams_from_disk = True,
              languages = None, window_size = 2 ** 10, freq_bands = 128,
              use_log10 = True):
    if load_grams_from_disk:
        if use_chromagrams:
            fp = "output/chromagrams.npz"
        else:
            fp = "output/melspectrograms.npz"
        with np.load(fp) as data:
            inputs = data["grams"]
            labels = data["labels"]
    else:
        # Loading CSV file
        lfilter_set = set(languages) if languages is not None else None
        print("Loading CSV file")
        audio_file_names = load_csv()
        sigs, srs, labels = process_audio_files(audio_file_names, lfilter=lfilter_set)

        # Create spectrograms or chromagrams
        print("Creating -grams")
        if use_chromagrams:
            chromagrams = get_chromagrams(sigs, srs, wsize = window_size,
                                          freq_bands = freq_bands,
                                          log10 = use_log10)
            inputs = chromagrams
            print(chromagrams.shape)
        else:
            mel_spectrograms = get_mel_spectrograms(sigs, srs, wsize = window_size,
                                                    freq_bands = freq_bands,
                                                    log10 = use_log10)
            inputs = mel_spectrograms
            print(mel_spectrograms.shape)
    return(inputs, labels)

#audio_file_names = load_csv()
#spectrograms, sample_rates, labels = get_mel_spectrograms(audio_file_names)
#print(len(audio_file_names))
#print(spectrograms.shape)
