from __future__ import division
import torch
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None

class MEL(object):
    """Create MEL Spectrograms from a raw audio signal. Relatively pretty slow.

       Usage (see librosa.feature.melspectrogram docs):
           MEL(sr=16000, n_fft=1600, hop_length=800, n_mels=64)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (samples x channels)

        Returns:
            tensor: (n_mels x hops x channels), where n_mels is the number of
                mel bins, hops is the number of hops, and channels is unchanged

        """
        if librosa is None:
            print("librosa not installed, cannot create spectrograms")
            return tensor
        L = []
        for i in range(tensor.size(1)):
            nparr = tensor[:, i].numpy() # (samples, )
            sgram = librosa.feature.melspectrogram(nparr, **self.kwargs) # (n_mels, hops)
            L.append(sgram)
        L = np.stack(L, 2) # (n_mels, hops, channels)
        tensor = torch.from_numpy(L).type_as(tensor)

        return tensor

class BLC2CBL(object):
    """Permute a 3d tensor from Bands x samples (Length) x Channels to Channels x
       Bands x samples (Length)
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

        Returns:
            tensor (Tensor): Tensor of spectrogram with shape (CxBxL)

        """

        return tensor.permute(2, 0, 1).contiguous()

class Dup(object):
    """Duplicate tensor to work with certain pretrained models that expect inputs
       of a certain minimum dimension.
    """

    def __init__(self, dups=2, dim=1):
        self.dups = dups
        self.dim = dim

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor (CxBxL)

        Returns:
            tensor (tensor): (Cxdups*BxL)

        """

        tensor = torch.cat([tensor]*self.dups, self.dim)

        return tensor


class LENC(object):
    """Transform labels into numerical representations.  This is a poor-man's
       version of the sklearn LabelEncoder.
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, s):
        """

        Args:
            s (str): string representing a langauge (i.e. "de")

        Returns:
            s_int (int): integer representation of s

        """

        return self.vocab[s]

class WC(object):
    """Transform Word Counter.
    """
    def __init__(self, exclude_vocab=None):
        self.exclude_vocab = exclude_vocab

    def __call__(self, s):
        """
        TODO add exclusion vocabulary

        Args:
            s (str): a string

        Returns:
            s_int (int): integer of the number of words in the string

        """

        return s.strip().count(" ")
