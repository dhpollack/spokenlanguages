import numpy as np
from load_data import *
import matplotlib.pyplot as plt

CMAP_COLOR = "jet"
IMG_SIZE = (4, 3)

audio_file_names = load_csv()
sigs, srs, labels = process_audio_files(audio_file_names)
mel_spectrograms = get_mel_spectrograms(sigs, srs)
chromagrams = get_chromagrams(sigs, srs)
#print(mel_spectrograms[0])
print(chromagrams.shape)

np.savez_compressed("output/sigs_srs_labels.npz", sigs=sigs, srs=srs, labels=labels)

label_set = set(labels)
for i, (sig, label) in enumerate(zip(sigs, labels)):
    if label in label_set:
        print(i, sig.shape, label)
        label_set.remove(label)
        fig1, ax1 = plt.subplots(figsize=IMG_SIZE)
        ax1.imshow(np.log10(mel_spectrograms[i]), interpolation='nearest',
                            aspect='auto', origin="lower", cmap=CMAP_COLOR)
        fig1.savefig("output/spectrogram_mel_"+label+".png")

        fig2, ax2 = plt.subplots(figsize=IMG_SIZE)
        ax2.imshow(chromagrams[i], interpolation='nearest', aspect='auto',
                   origin="lower", cmap=CMAP_COLOR)
        fig2.savefig("output/chromagram_"+label+".png")
    if len(label_set) == 0: break
