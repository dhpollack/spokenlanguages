import argparse
import numpy as np
from load_data import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models import *

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Basic 2-Layer Language ID Classifier')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=8, metavar='b',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--file-list', type=str, default="data/testingset.csv",
                    help='csv file with audio files and labels')
parser.add_argument('--use-chromagrams', action='store_true',
                    help='use chromagrams')
parser.add_argument('--load-grams', action='store_true',
                    help='load spectro/chroma -grams')
parser.add_argument('--model', type=str, default="cnn",
                    help='csv file with audio files and labels')
parser.add_argument('--model-path', type=str, default="output/states/cnn_model_spectra.pt",
                    help='csv file with audio files and labels')
args = parser.parse_args()


# set seed
torch.manual_seed(args.seed)

# load spectro/chroma -grams
inputs, labels = get_grams(filelist = args.file_list,
                           use_chromagrams = args.use_chromagrams,
                           load_grams_from_disk = args.load_grams)
le = LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)
num_targets = le.classes_.shape[0]

models_dict = {"cnn": cnn.Net(num_targets), "resnet": resnet.resnetX(pretrained=False, layers=[2, 2, 2, 2], num_classes=num_targets)}

# Create Network
net = models_dict[args.model]
net.load_state_dict(torch.load(args.model_path))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
print(net)
def transform(tensor):
    if args.model == "cnn":
        return(Variable(tensor.unsqueeze(1)))

# make predictions in batches
output = []
for t in torch.split(torch.Tensor(inputs), args.batch_size):
    output += [net(transform(t)).data]
output = torch.cat(output)
yhat = le.inverse_transform(output.max(1)[1].numpy())
print("Accuracy: %.2f"%accuracy_score(labels, yhat))
print(confusion_matrix(labels, yhat))
