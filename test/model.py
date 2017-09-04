import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
import models.resnet
import models.squeezenet
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import spl_transforms
from v2_data_fetch import *

class Test_Squeezenet(unittest.TestCase):

    bdir = "data/voxforge"

    def test1(self):
        # Data
        vx = VOXFORGE(self.bdir, label_type="lang")
        vx.find_max_len()
        print(vx.maxlen)
        T = tat.Compose([
                tat.PadTrim(vx.maxlen),
                spl_transforms.MEL(n_mels=224),
                spl_transforms.BLC2CBL(),
                tvt.ToPILImage(),
                tvt.Scale((224, 224)),
                tvt.ToTensor(),
            ])
        TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        dl = data.DataLoader(vx, batch_size = 25, shuffle=True)

        # Model and Loss
        model = models.squeezenet.squeezenet(True)
        model.train()

        for i, (mb, tgts) in enumerate(dl):
            vx.set_split("train")
            out = model(Variable(mb))
            print(mb.size(), mb.min(), mb.max())
            print(out.data.size())
            print(out.data)
            break

    def test2(self):
        # Data
        vx = VOXFORGE(self.bdir, label_type="lang")
        vx.find_max_len()
        #vx.maxlen = 150000
        T = tat.Compose([
                tat.PadTrim(vx.maxlen),
                spl_transforms.MEL(n_mels=224),
                spl_transforms.BLC2CBL(),
                tvt.ToPILImage(),
                tvt.Scale((224, 224)),
                tvt.ToTensor(),
            ])
        TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        dl = data.DataLoader(vx, batch_size = 25, shuffle=True)

        # Model and Loss
        model = models.squeezenet.squeezenet(True)
        criterion = nn.CrossEntropyLoss()
        plist = nn.ParameterList()
        #plist.extend(list(model[0].parameters()))
        plist.extend(list(model[1].classifier.parameters()))
        #plist.extend(list(model.parameters()))
        #optimizer = torch.optim.SGD(plist, lr=0.0001, momentum=0.9)
        optimizer = torch.optim.Adam(plist, lr=0.0001)

        train_losses = []
        valid_losses = []
        for i, (mb, tgts) in enumerate(dl):
            model.train()
            vx.set_split("train")
            mb, tgts = Variable(mb), Variable(tgts)
            model.zero_grad()
            out = model(mb)
            loss = criterion(out, tgts)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.data[0])
            print(loss.data[0])
            if i % 5 == 0:
                model.eval()
                vx.set_split("valid")
                running_validation_loss = 0
                correct = 0
                for mb_valid, tgts_valid in dl:
                    mb_valid, tgts_valid = Variable(mb_valid), Variable(tgts_valid)
                    out_valid = model(mb_valid)
                    loss_valid = criterion(out_valid, tgts_valid)
                    running_validation_loss += loss_valid.data[0]
                    correct += (out_valid.data.max(1)[1] == tgts_valid.data).sum()
                valid_losses.append((running_validation_loss, correct / len(vx)))
                print("loss: {}, acc: {}".format(running_validation_loss, correct / len(vx)))

class Test_Resnet(unittest.TestCase):

    bdir = "data/voxforge"

    def test1(self):
        # Data
        vx = VOXFORGE(self.bdir, label_type="lang")
        #vx.find_max_len()
        vx.maxlen = 150000
        T = tat.Compose([
                tat.PadTrim(vx.maxlen),
                tat.MEL(n_mels=224),
                tat.BLC2CBL(),
                tvt.ToPILImage(),
                tvt.Scale((224, 224)),
                tvt.ToTensor(),
            ])
        TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        dl = data.DataLoader(vx, batch_size = 25, shuffle=True)

        # Model and Loss
        model = models.resnet.resnet34(True)
        print(model)
        criterion = nn.CrossEntropyLoss()
        plist = nn.ParameterList()
        #plist.extend(list(model[0].parameters()))
        plist.extend(list(model[1].fc.parameters()))
        #plist.extend(list(model.parameters()))
        #optimizer = torch.optim.SGD(plist, lr=0.0001, momentum=0.9)
        optimizer = torch.optim.Adam(plist, lr=0.0001)

        train_losses = []
        valid_losses = []
        for i, (mb, tgts) in enumerate(dl):
            model.train()
            vx.set_split("train")
            mb, tgts = Variable(mb), Variable(tgts)
            model.zero_grad()
            out = model(mb)
            loss = criterion(out, tgts)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.data[0])
            print(loss.data[0])
            if i % 5 == 0:
                model.eval()
                vx.set_split("valid")
                running_validation_loss = 0
                correct = 0
                for mb_valid, tgts_valid in dl:
                    mb_valid, tgts_valid = Variable(mb_valid), Variable(tgts_valid)
                    out_valid = model(mb_valid)
                    loss_valid = criterion(out_valid, tgts_valid)
                    running_validation_loss += loss_valid.data[0]
                    correct += (out_valid.data.max(1)[1] == tgts_valid.data).sum()
                valid_losses.append((running_validation_loss, correct / len(vx)))
                print("loss: {}, acc: {}".format(running_validation_loss, correct / len(vx)))
            break

if __name__ == '__main__':
    unittest.main()
