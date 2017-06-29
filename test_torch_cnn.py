import argparse
import numpy as np
from load_data import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import models.cnn as model

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Basic 2-Layer Language ID Classifier')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=8, metavar='b',
                    help='batch size')
parser.add_argument('--languages', type=str, nargs='+', default=None,
                    help='languages to filter by')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                    help='reports per epoch')
parser.add_argument('--file-list', type=str, default="data/trainingset.csv",
                    help='csv file with audio files and labels')
parser.add_argument('--load-model', type=bool,  default=True,
                    help='load model from disk')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

lfilter_set = set(args.languages) if args.languages is not None else None

audio_file_names = load_csv()
sigs, srs, labels = process_audio_files(audio_file_names, lfilter=lfilter_set)
le = LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)
num_targets = le.classes_.shape[0]

# Create spectrograms or chromagrams
window_size = 2 ** 10
use_spectrograms = True
if use_spectrograms:
    mel_spectrograms = get_mel_spectrograms(sigs, srs, wsize = window_size)
    inputs = mel_spectrograms
    print(mel_spectrograms.shape)
else:
    chromagrams = get_chromagrams(sigs, srs, wsize = window_size)
    inputs = chromagrams
    print(chromagrams.shape)


# Create Network
model_save_path = "output/states/cnn_model_state.pt"
use_saved_model = False
net = model.Net(num_targets)
if use_saved_model:
    torch.load(model_save_path)
    #net.load_state_dict(model_save_path)
print(net)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


# create training and testing sets
train_frac = 0.8
train_idx = int(len(inputs) * train_frac)
inputs_training = inputs[:train_idx]
labels_training = labels_encoded[:train_idx]
inputs_testing = inputs[train_idx:]
labels_testing = labels_encoded[train_idx:]

# calculate number of minibatches
make3d = True
b = args.batch_size
minibatches = int(train_idx / b)
print_freq = minibatches // args.log_interval
print("Minibatches:", minibatches)
print("Print Frequency:", print_freq)

def get_minibatch(X, y, n, b, make3d = False, rc_len = None):
    # get the inputs+labels
    end_idx = n+b if (n+b) <= len(y) else len(y)
    mb = X[n:end_idx,:,:]
    l = y[n:end_idx]
    # get random cut
    if rc_len is not None:
        cut_start = np.random.randint(X.shape[2]-rc_len)
        cut_end = cut_start + rc_len
        mb = mb[:,:,cut_start:cut_end]
    # wrap them in Variable
    mb, l = torch.from_numpy(mb).float(), torch.from_numpy(l)
    if make3d:
        mb.unsqueeze_(1)
    mb, l = Variable(mb), Variable(l)
    n += b
    return(mb, l, n)

# per epoch variables
epochs = args.epochs
shuffle_inputs = True

for epoch in range(epochs):  # loop over the dataset multiple times
    # reset vars each epoch
    n = 0
    running_loss = 0.0
    if shuffle_inputs:
        idx_shf = np.random.permutation(inputs_training.shape[0])
        inputs_training, labels_training = inputs_training[idx_shf], labels_training[idx_shf]
    for i in range(minibatches):
        # get minibatch
        minibatch, l, n = get_minibatch(inputs_training, labels_training, n, b, make3d = make3d)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(minibatch)
        loss = criterion(outputs, l)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

        if i % print_freq == (print_freq-1):
            print('[%d, %5d, %d] loss: %.5f' %
                  (epoch + 1, i + 1, n, running_loss / n)) # average loss in epoch
            #running_loss = 0.0 # uncomment to get loss per print

print('Finished Training')

inputs_tensored = torch.from_numpy(inputs_testing).float()
if make3d:
    inputs_tensored.unsqueeze_(1)
outputs = net(Variable(inputs_tensored))
outputs_labels = outputs.max(1)[1].data.numpy().ravel()
print(outputs_labels.shape)

yhat = le.inverse_transform(outputs_labels)
print(yhat)
y_t = le.inverse_transform(labels_testing)
print(y_t)
print(accuracy_score(y_t, yhat))
print(confusion_matrix(y_t, yhat))
torch.save(net.state_dict(), model_save_path)
