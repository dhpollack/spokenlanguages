import numpy as np
import librosa

print("Beginning test...")

filelist = "data/trainingset.csv"

l = []
with open(filelist, "r") as f:
    for line in f.readlines():
        fp, lab = line.strip().split(",")
        l.append([fp, lab])
print(l[0][0], l[0][1])

filename = "data/train/" + l[0][0]


# librosa
print("librosa:")
sig1, sr1 = librosa.core.load(filename, sr=None)

print(sig1.shape, sr1)
print(sig1.max(), sig1.min(), sig1.mean(), sig1.std())
