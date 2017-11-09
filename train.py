import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import models.resnet
import models.squeezenet
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import spl_transforms
from loader_voxforge import *

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=100, metavar='b',
                    help='batch size')
parser.add_argument('--freq-bands', type=int, default=224,
                    help='number of frequency bands to use')
parser.add_argument('--languages', type=str, nargs='+', default=["de", "en", "es"],
                    help='languages to filter by')
parser.add_argument('--data-path', type=str, default="data/voxforge",
                    help='data path')
parser.add_argument('--use-cache', action='store_true',
                    help='use cache in the dataloader')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for data loader')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
parser.add_argument('--train-full-model', action='store_true',
                    help='train full model vs. final layer')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

# use_cuda
use_cuda = torch.cuda.is_available()
print("CUDA: {}".format(use_cuda))

# Data
vx = VOXFORGE(args.data_path, langs=args.languages, label_type="lang")
#vx.find_max_len()
vx.maxlen = 150000
T = tat.Compose([
        tat.PadTrim(vx.maxlen),
        tat.MEL(n_mels=args.freq_bands),
        tat.BLC2CBL(),
        tvt.ToPILImage(),
        tvt.Scale((args.freq_bands, args.freq_bands)),
        tvt.ToTensor(),
    ])
TT = spl_transforms.LENC(vx.LABELS)
vx.transform = T
vx.target_transform = TT
dl = data.DataLoader(vx, batch_size=args.batch_size,
                     num_workers=args.num_workers, shuffle=True)

# Model and Loss
model = models.resnet.resnet34(True, num_langs=5)
if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))
print(model)
criterion = nn.CrossEntropyLoss()
plist = nn.ParameterList()
if args.train_full_model:
    #plist.extend(list(model[0].parameters()))
    plist.extend(list(model.parameters()))
    optimizer = torch.optim.SGD(plist, lr=args.lr, momentum=0.9)
else:
    plist.extend(list(model[1].fc.parameters()))
    optimizer = torch.optim.Adam(plist, lr=args.lr)

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
        if i % args.log_interval == 0:
            validate(epoch)
            if args.save_model:
                torch.save(model.state_dict(), "output/states/model_resnet34_{}.pt".format(epoch+1))
        vx.set_split("train")

def validate(epoch):
    model.eval()
    vx.set_split("valid")
    running_validation_loss = 0
    correct = 0
    for mb_valid, tgts_valid in dl:
        if use_cuda:
            mb_valid, tgts_valid = mb_valid.cuda(), tgts_valid.cuda()
        mb_valid, tgts_valid = Variable(mb_valid), Variable(tgts_valid)
        out_valid = model(mb_valid)
        loss_valid = criterion(out_valid, tgts_valid)
        running_validation_loss += loss_valid.data[0]
        correct += (out_valid.data.max(1)[1] == tgts_valid.data).sum()
    valid_losses.append((running_validation_loss, correct / len(vx)))
    print("loss: {}, acc: {}".format(running_validation_loss, correct / len(vx)))


epochs = args.epochs
train_losses = []
valid_losses = []
for epoch in range(epochs):
    print("epoch {}".format(epoch + 1))
    train(epoch)
    if args.save_model:
        torch.save(model.state_dict(), "output/states/model_resnet34_{}.pt".format(epoch+1))
