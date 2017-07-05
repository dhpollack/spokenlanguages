import argparse
from load_data import *
import numpy

parser = argparse.ArgumentParser(description='Create spectrograms and chromagrams')
parser.add_argument('--use-log10', action='store_true',
                    help='use log of grams')
parser.add_argument('--window-size', type=int, default=2**10,
                    help='window size')
parser.add_argument('--bands', type=int, default=128,
                    help='bands')
parser.add_argument('--languages', type=str, nargs='+', default=None,
                    help='languages to filter by')
parser.add_argument('--file-name-prefix', type=str, default="",
                    help='prefix for filename')
args = parser.parse_args()

# Loading CSV file
lfilter_set = set(args.languages) if args.languages is not None else None
print("Loading CSV file")
audio_file_names = load_csv()
sigs, srs, labels = process_audio_files(audio_file_names, lfilter=lfilter_set)

chromagrams = get_chromagrams(sigs, srs, wsize = args.window_size,
                              freq_bands = args.bands, log10 = args.use_log10)
mel_spectrograms = get_mel_spectrograms(sigs, srs, wsize = args.window_size,
                                        freq_bands = args.bands, log10 = args.use_log10)

np.savez_compressed("output/" + args.file_name_prefix + "chromagrams.npz", grams=chromagrams, labels=labels)
np.savez_compressed("output/" + args.file_name_prefix + "melspectrograms.npz", grams=mel_spectrograms, labels=labels)
