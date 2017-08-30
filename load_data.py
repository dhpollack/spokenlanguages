import numpy as np
import os
import librosa

def load_csv(filelist = "data/trainingset.csv"):
    l = []
    with open(filelist, "r") as f:
        for line in f.readlines():
            fp, lab = line.strip().split(",")
            l.append([fp, lab])
    return(l)

def get_audio_paths(fp):
    dn = os.path.dirname(fp)
    bn = os.path.basename(fp)
    return(os.path.join(dn, bn.split('.')[0]))

def process_audio_files(filelist = "data/trainingset.csv", N=None,
                        base_sr = 44100, lfilter = None):
    basedir = get_audio_paths(filelist)
    sigs = []
    srs = []
    labels = []
    l = load_csv(filelist)
    l = l[:N] if l is not None else l
    files = []
    for fname, label in l:
        if (lfilter == None) or (label in lfilter):
            filename = os.path.join(basedir, fname)
            sig, sr = librosa.core.load(filename, sr=None)
            if base_sr == None or sr == base_sr:
                sig = np.trim_zeros(sig)
                sigs.append(sig)
                srs.append(sr)
                labels.append(label)
                files.append((fname,label))
    np.savetxt("/tmp/filelist.csv", delimiter=",", fmt="%s")
    return(sigs, srs, labels)

def create_grams(sigs, srs, base_sr = 44100, wsize = 2**10, freq_bands = 128,
                 gram_type = "spectrograms", normalize = True, verbose = True):
    hsize = wsize//2
    grams = []
    shapes = set()
    base_sr = base_sr if np.unique(srs).shape[0] > 1 else np.unique(srs)[0]
    if gram_type == "spectrograms":
        kargs = {"n_mels":freq_bands}
        gram_func = librosa.feature.melspectrogram
    elif gram_type == "chromagrams":
        kargs = {"n_chroma":freq_bands}
        gram_func = librosa.feature.chroma_stft
    for sig, sr in zip(sigs, srs):
        if sr == base_sr:
            gram = gram_func(sig, sr=sr, n_fft=wsize, hop_length=hsize, **kargs)
            if normalize: # normalization
                gram -= gram.mean()
                gram /= np.abs(gram).max()
            shapes.add(gram.shape)
            grams.append(gram)
        else:
            print(sr)
    min_shape = min(shapes)
    grams = np.array([x[:,:min_shape[1]] for x in grams])
    #grams /= grams.max()
    #grams = np.log10(grams)
    if verbose:
        win_times = 1000 * wsize / base_sr
        hop_times = 1000 * hsize / base_sr
        print('Window Length: %.2fms'%win_times)
        print('Hop Length: %.2fms'%hop_times)
    return(grams)

def get_grams(filelist = "data/trainingset.csv", N = None, languages = None,
              use_chromagrams = False, grams_path = None,
              base_sr = 44100, window_size = 2 ** 10, freq_bands = 128):
    if grams_path is not None:
        with np.load(grams_path) as data:
            inputs = data["grams"]
            labels = data["labels"]
    else:
        # Loading CSV file
        lfilter_set = set(languages) if languages is not None else None
        print("Loading CSV file + Audio Files")
        sigs, srs, labels = process_audio_files(filelist, N=N,
                                                base_sr = base_sr,
                                                lfilter=lfilter_set)
        # Create spectrograms or chromagrams
        print("Creating -grams")
        if use_chromagrams:
            gram_type = "chromagrams"
        else:
            gram_type = "spectrograms"
        inputs = create_grams(sigs, srs, gram_type = gram_type, base_sr = base_sr,
                              wsize = window_size, freq_bands = freq_bands)
    print(inputs.shape)
    return(inputs, labels)
