import argparse
from load_data import *
import numpy

parser = argparse.ArgumentParser(description='Create spectrograms and chromagrams')
parser.add_argument('--use-log10', action='store_true',
                    help='use log of grams')
parser.add_argument('--languages', type=str, nargs='+', default=None,
                    help='languages to filter by')
args = parser.parse_args()

# Loading CSV file
lfilter_set = set(args.languages) if args.languages is not None else None
print("Loading CSV file")
audio_file_names = load_csv()
sigs, srs, labels = process_audio_files(audio_file_names, lfilter=lfilter_set)

window_size = 2 ** 10
chromagrams = get_chromagrams(sigs, srs, wsize = window_size, log10 = args.use_log10)
mel_spectrograms = get_mel_spectrograms(sigs, srs, wsize = window_size, log10 = args.use_log10)

np.savez_compressed("output/chromagrams.npz", grams=chromagrams, labels=labels)
np.savez_compressed("output/melspectrograms.npz", grams=mel_spectrograms, labels=labels)
