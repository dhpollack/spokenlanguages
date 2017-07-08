import argparse
from load_data import *
import numpy

parser = argparse.ArgumentParser(description='Create spectrograms and chromagrams')
parser.add_argument('--filelist', type=str, default="data/trainingset.csv",
                    help='location of csv file with file names')
parser.add_argument('--window-size', type=int, default=2**11,
                    help='window size')
parser.add_argument('--freq-bands', type=int, default=224,
                    help='number of frequency bands to use')
parser.add_argument('--languages', type=str, nargs='+', default=None,
                    help='languages to filter by')
parser.add_argument('--file-name-prefix', type=str, default="",
                    help='prefix for filename')
args = parser.parse_args()

# Loading CSV file
lfilter_set = set(args.languages) if args.languages is not None else None
print("Loading CSV file")
sigs, srs, labels = process_audio_files(args.filelist, lfilter=lfilter_set)

S = create_grams(sigs, srs, gram_type = "spectrograms",
                 wsize = args.window_size, freq_bands = args.freq_bands)
C = create_grams(sigs, srs, gram_type = "chromagrams",
                 wsize = args.window_size, freq_bands = args.freq_bands)

np.savez_compressed("output/" + args.file_name_prefix + "spectrograms.npz", grams=S, labels=labels)
np.savez_compressed("output/" + args.file_name_prefix + "chromagrams.npz", grams=C, labels=labels)
