import torch
import torch.nn as nn
from torch.autograd import Variable
import models.resnet
import models.squeezenet
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import spl_transforms
from loader_voxforge import *

# use_cuda
use_cuda = torch.cuda.is_available()

# Data
vx = VOXFORGE("data/voxforge", langs=["de", "en", "sp"], label_type="lang")
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
model = models.resnet.resnet34(True, num_langs=5)
print(model)
criterion = nn.CrossEntropyLoss()
plist = nn.ParameterList()
#plist.extend(list(model[0].parameters()))
plist.extend(list(model[1].fc.parameters()))
#plist.extend(list(model.parameters()))
#optimizer = torch.optim.SGD(plist, lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(plist, lr=0.0001)

if use_cuda:
    model = model.cuda()

def train(epoch):
    vx.set_split("train")
    for i, (mb, tgts) in enumerate(dl):
        model.train()
        if use_cuda:
            mb, tgts = mb.cuda(), tgts.cuda()
        mb, tgts = Variable(mb), Variable(tgts)
        model.zero_grad()
        out = model(mb)
        loss = criterion(out, tgts)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.data[0])
        print(loss.data[0])
        if i % 5 == 0:
            validate(epoch)
        vx.set_split("train")

def validate(epoch):
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


epochs = 10
train_losses = []
valid_losses = []
for epoch in range(epochs):
    train(epoch)
    model.save_state_dict("output/states/model_resnet34_{}.pt".format(epoch+1))
