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

class DummyDim(object):
    """add a dummy dimension
    """
    def __init__(self, dim=-1):
        self.dim = dim
    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor without dummy time

        Returns:
            tensor (Tensor): Tensor with dummy dim

        """

        return tensor.unsqueeze(self.dim)

class SqueezeDim(object):
    """squeeze out a dimension
    """
    def __init__(self, dim=-1):
        self.dim = dim
    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor with extra dim

        Returns:
            tensor (Tensor): Tensor without dim

        """

        return tensor.squeeze(self.dim)

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

class Preemphasis(object):
    """Perform preemphasis on signal

    y = x[n] - α*x[n-1]

    Args:
        alpha (float): preemphasis coefficient

    """

    def __init__(self, alpha=0.97):
        self.alpha = alpha

    def __call__(self, sig):
        """

        Args:
            sig (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            sig (Tensor): Preemphasized. See equation above.

        """
        if self.alpha == 0:
            return sig
        else:
            sig[1:, :] -= self.alpha * sig[:-1, :]
            return sig

class RfftPow(object):
    """This function emulates power of the discrete fourier transform.

    Note: this implementation may not be numerically stable

    Args:
        K (int): number of fft freq bands

    """

    def __init__(self, K=None):
        self.K = K

    def __call__(self, sig):
        """

        Args:
            sig (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            S (Tensor): spectrogram

        """
        N = sig.size(1)
        if self.K is None:
            K = N
        else:
            K = self.K

        k_vec = torch.arange(0, K).unsqueeze(0)
        n_vec = torch.arange(0, N).unsqueeze(1)
        angular_pt = 2 * np.pi * k_vec * n_vec / K
        S = torch.sqrt(torch.matmul(sig, angular_pt.cos())**2 + \
                       torch.matmul(sig, angular_pt.sin())**2)
        S = S.squeeze()[:(K//2+1)]
        S = (1 / K) * S**2
        return S

class FilterBanks(object):
    """Bins a periodogram from K fft frequency bands into N bins (banks)

    fft bands (K//2+1) -> filterbanks (n_filterbanks) -> bins (bins)

    Args:
        n_filterbanks (int): number of filterbanks
        bins (list): number of bins

    """

    def __init__(self, n_filterbanks, bins):
        self.n_filterbanks = n_filterbanks
        self.bins = bins

    def __call__(self, S):
        """

        Args:
            S (Tensor): Tensor of Spectro- / Periodogram

        Returns:
            fb (Tensor): binned filterbanked spectrogram

        """
        conversion_factor = np.log(10) # torch.log10 doesn't exist
        K = S.size(0)
        fb_mat = torch.zeros((self.n_filterbanks, K))
        for m in range(1, self.n_filterbanks+1):
            f_m_minus = int(self.bins[m - 1])
            f_m = int(self.bins[m])
            f_m_plus = int(self.bins[m + 1])

            fb_mat[m - 1, f_m_minus:f_m] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            fb_mat[m - 1, f_m:f_m_plus] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
        fb = torch.matmul(S, fb_mat.t())
        fb = 20 * torch.log(fb) / conversion_factor
        return fb

class MFCC(object):
    """Discrete Cosine Transform

    There are three types of the DCT.  This is 'Type 2' as described in the scipy docs.

    filterbank bins (bins) -> mfcc (mfcc)

    Args:
        n_filterbanks (int): number of filterbanks
        n_coeffs (int): number of mfc coefficients to keep
        mode (str): orthogonal transformation

    """

    def __init__(self, n_filterbanks, n_coeffs, mode="ortho"):
        self.n_filterbanks = n_filterbanks
        self.n_coeffs = n_coeffs
        self.mode = "ortho"

    def __call__(self, fb):
        """

        Args:
            fb (Tensor): Tensor of binned filterbanked spectrogram

        Returns:
            mfcc (Tensor): Tensor of mfcc coefficients

        """
        K = self.n_filterbanks
        k_vec = torch.arange(0, K).unsqueeze(0)
        n_vec = torch.arange(0, self.n_filterbanks).unsqueeze(1)
        angular_pt = np.pi * k_vec * ((2*n_vec+1) / (2*K))
        mfcc = 2 * torch.matmul(fb, angular_pt.cos())
        if self.mode == "ortho":
            mfcc[0] *= np.sqrt(1/(4*self.n_filterbanks))
            mfcc[1:] *= np.sqrt(1/(2*self.n_filterbanks))
        return mfcc[1:(self.n_coeffs+1)]

class Sig2Features(object):
    """Get the log power, MFCCs and 1st derivatives of the signal across n hops
    and concatenate all that together

    Args:
        n_hops (int): number of filterbanks
        transformDict (dict): dict of transformations for each hop

    """

    def __init__(self, ws, hs, transformDict):
        self.ws = ws
        self.hs = hs
        self.td = transformDict

    def __call__(self, sig):
        """

        Args:
            sig (Tensor): Tensor of signal

        Returns:
            Feats (Tensor): Tensor of log-power, 12 mfcc coefficients and 1st devs

        """
        n_hops = (sig.size(0) - self.ws) // self.hs

        P = []
        Mfcc = []

        for i in range(n_hops):
            # create frame
            st = int(i * self.hs)
            end = st + self.ws
            sig_n = sig[st:end]

            # get power/energy
            P += [self.td["RfftPow"](sig_n.transpose(0, 1))]

            # get mfccs and filter banks
            fb = self.td["FilterBanks"](P[-1])
            Mfcc += [self.td["MFCC"](fb)]

        # concat and calculate derivatives
        P = torch.stack(P, 1)
        P_sum = torch.log(P.sum(0))
        P_dev = torch.zeros(P_sum.size())
        P_dev[1:] = P_sum[1:] - P_sum[:-1]
        Mfcc = torch.stack(Mfcc, 1)
        Mfcc_dev = torch.cat((torch.zeros(self.td["MFCC"].n_coeffs, 1), Mfcc[:,:-1] - Mfcc[:,1:]), 1)
        Feats = torch.cat((P_sum.unsqueeze(0), P_dev.unsqueeze(0), Mfcc, Mfcc_dev), 0)
        return Feats

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
