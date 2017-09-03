import unittest
import torchaudio.transforms as transforms
import spl_transforms
from v2_data_fetch import *

class Test_Voxforge(unittest.TestCase):
    bdir = "data/voxforge"

    def test1(self):
        vx = VOXFORGE(self.bdir, download=False, label_type="lang",
                      num_zips=10, randomize=False,
                      dev_mode=True)
        for i, (a, b) in enumerate(vx):
            print(a.size(), b)
            if i > 10: break

    def test2(self):
        vx = VOXFORGE(self.bdir, download=False, label_type="lang",
                      num_zips=10, randomize=False,
                      dev_mode=True)
        maxlen = 0
        for i, (a, b) in enumerate(vx):
            maxlen = a.size(0) if a.size(0) > maxlen else maxlen
        print(maxlen)
        vx.find_max_len()
        self.assertEqual(maxlen, vx.maxlen)

    def test3(self):
        vx = VOXFORGE(self.bdir, download=False, label_type="lang",
                      num_zips=10, randomize=False, split="valid",
                      dev_mode=False)
        vx.find_max_len()
        T = transforms.Compose([transforms.PadTrim(vx.maxlen),
                                spl_transforms.MEL(),
                                spl_transforms.BLC2CBL()])
        TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        dl = data.DataLoader(vx, batch_size = 5)
        labels_total = 0
        for i, (a, b) in enumerate(dl):
            labels_total += b.sum()
        print((len(vx)-labels_total) / len(vx))

    def test4(self):
        vx = VOXFORGE(self.bdir, download=False, label_type="lang",
                      num_zips=10, randomize=False,
                      dev_mode=False)
        vx.find_max_len()
        T = transforms.Compose([transforms.PadTrim(vx.maxlen),])
        TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        print(vx.splits)
        dl = data.DataLoader(vx, batch_size = 5)
        total_train = 0
        for i, (mb, l) in enumerate(dl):
            vx.set_split("train")
            total_train += l.size(0)
            if i == 2:
                vx.set_split("valid")
                total_valid = 0
                for mb_valid, l_valid in dl:
                    total_valid += l_valid.size(0)
                print(total_valid)
        print(total_train)

if __name__ == '__main__':
    unittest.main()
