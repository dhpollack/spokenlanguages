import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import models
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import spl_transforms
from loader_voxforge import *
import cfg

import csv

def create_optimizer(optim, params, kwargs):
    plist = nn.ParameterList(list(params))
    return optim(plist, **kwargs)

def get_optimizer(epoch):
    grenz = 0
    for i, (k, v) in enumerate(cfg_train["epochs"]):
        grenz += v
        if epoch == 0:
            print(k, epoch, grenz)
            opt = cfg_train[k]["optimizer"]
            params = cfg_train[k]["params"]
            kwargs = cfg_train[k]["optim_kwargs"]
            print("Using new optimizer: {} with args {}".format(opt, kwargs))
            return create_optimizer(opt, params, kwargs)
        elif epoch == grenz:
            print(k, epoch, grenz)
            k_next, _ = cfg_train["epochs"][i+1]
            opt = cfg_train[k_next]["optimizer"]
            params = cfg_train[k_next]["params"]
            kwargs = cfg_train[k_next]["optim_kwargs"]
            print("Using new optimizer: {} with args {}".format(opt, kwargs))
            return create_optimizer(opt, params, kwargs)
        else:
            pass
    return optimizer

def train(epoch):
    vx.set_split("train")
    global optimizer
    optimizer = get_optimizer(epoch)
    epoch_losses = []
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
        epoch_losses.append(loss.data[0])
        print(loss.data[0])
        if i % args.log_interval == 0 and args.validate and i != 0:
            validate(epoch)
        vx.set_split("train")
    train_losses.append(epoch_losses)

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

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--batch-size', type=int, default=100, metavar='b',
                    help='batch size')
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
parser.add_argument('--chkpt-interval', type=int, default=10,
                    help='how often to save checkpoints')
parser.add_argument('--model-name', type=str, default="resnet34",
                    help='data path')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

# use_cuda
use_cuda = torch.cuda.is_available()
print("CUDA: {}".format(use_cuda))

# Data
vx = VOXFORGE(args.data_path, langs=args.languages,
              label_type="lang", use_cache=args.use_cache)
#vx.find_max_len()

# Data Loader and Model Configuration
cfg_model = cfg.MODELS[args.model_name]
T = cfg_model["T"]
vx.transform = T
TT = spl_transforms.LENC(vx.LABELS)
vx.target_transform = TT
dl = data.DataLoader(vx, batch_size=args.batch_size,
                     num_workers=args.num_workers, shuffle=True)

# Model and Loss Initializations
model = cfg_model["model"]
if args.load_model is not None:
    model.load_state_dict(torch.load(cfg_model["state_dict_path"]))
model = model.cuda() if use_cuda else model
print(model)
cfg_train = cfg.TRAINING[args.model_name]
criterion = cfg_train["criterion"]
optimizer = None

# Train
epochs = sum([v for (k, v) in cfg_train["epochs"]])
train_losses = []
valid_losses = []
for epoch in range(epochs):
    print("epoch {}".format(epoch + 1))
    train(epoch)
    if args.save_model and (epoch % args.chkpt_interval == 0 or epoch+1 == epochs):
        torch.save(model.state_dict(), "output/states/{}_{}.pt".format(args.model_name, epoch+1))
with open("output/train_losses_{}.csv".format(args.model_name), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(train_losses)
