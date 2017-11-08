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

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Predicter')
parser.add_argument('--batch-size', type=int, default=100, metavar='b',
                    help='batch size')
parser.add_argument('--freq-bands', type=int, default=224,
                    help='number of frequency bands to use')
parser.add_argument('--languages', type=str, nargs='+', default=["de", "en", "es"],
                    help='languages to filter by')
parser.add_argument('--data-path', type=str, default="data/voxforge",
                    help='data path')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--model-name', type=str, default=None,
                    help='name of model to load')
parser.add_argument('--model-path', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

# use_cuda
use_cuda = torch.cuda.is_available()
print("CUDA: {}".format(use_cuda))

# Data
vx = VOXFORGE(args.data_path, langs=args.languages, label_type="lang", use_cache=False)
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
dl = data.DataLoader(vx, batch_size = args.batch_size, shuffle=True)

model_name = args.model_name
model_path = args.model_path
use_pretrained = model_path is None

MODELS = {
    "resnet34": {
        "model": models.resnet.resnet34(use_pretrained, num_langs=5),
        "state_dict": model_path,
    },
    "squeezenet": {
        "model": models.squeezenet.squeezenet(use_pretrained, num_langs=5),
        "state_dict": model_path,
    }
}

# Model and Loss
if model_name not in MODELS:
    model_name = "resnet34"
model = MODELS[model_name]["model"]
if not use_pretrained:
    model.load_state_dict(torch.load(MODELS[model_name]["state_dict"], map_location=lambda storage, loc: storage))
print(model)
criterion = nn.CrossEntropyLoss()

if use_cuda:
    model = model.cuda()

def predict():
    model.eval()
    vx.set_split("test")
    running_test_loss = 0
    correct = 0
    for i, (mb_test, tgts_test) in enumerate(dl):
        if use_cuda:
            mb_test, tgts_test = mb_test.cuda(), tgts_test.cuda()
        mb_test, tgts_test = Variable(mb_test), Variable(tgts_test)
        out_test = model(mb_test)
        predictions = out_test.data.max(1)[1]
        if i % 10 == 0:
            print(predictions, tgts_test.data)
        loss_test = criterion(out_test, tgts_test)
        running_test_loss += loss_test.data[0]
        correct += (predictions == tgts_test.data).sum()
    print("loss: {}, acc: {}".format(running_test_loss, correct / len(vx)))

predict()
