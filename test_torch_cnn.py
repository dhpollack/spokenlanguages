import numpy as np
from load_data import *
from sklearn.preprocessing import LabelEncoder

#data = np.load("output/sigs_srs_labels.npz")
#sigs, srs, labels = data["sigs"], data["srs"], data["labels"]
#data = None

audio_file_names = load_csv()
sigs, srs, labels = process_audio_files(audio_file_names, N=600, lfilter=set(["English", "German", "Spanish"]))
le = LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)
num_targets = le.classes_.shape[0]

window_size = 2 ** 10

chromagrams = get_chromagrams(sigs, srs, wsize = window_size)
mel_spectrograms = get_mel_spectrograms(sigs, srs, wsize = window_size)

three_secs = int(3.0 / (window_size//2 / srs[0]))

print(three_secs)
print(chromagrams.shape)
print(mel_spectrograms.shape)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_targets):
        super(Net, self).__init__()
        self.features_out_conv = 97440 # dummy value
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(self.features_out_conv, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_targets)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net(num_targets)
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

inputs = mel_spectrograms # or chromagrams

b = 4
epochs = 1
train_frac = 0.8
minibatches = int(len(labels)*train_frac / b)
print(minibatches)

print_freq = minibatches // 4
#minibatches = 25

def get_minibatch(X, y, n, b, rc_len = None):
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
    mb, l = Variable(mb), Variable(l)
    n += b
    return(mb, l, n)

for epoch in range(epochs):  # loop over the dataset multiple times
    # reset vars each epoch
    n = 0
    running_loss = 0.0
    for i in range(minibatches):
        # get minibatch
        minibatch, l, n = get_minibatch(inputs, labels_encoded, n, b)
        # zero the parameter gradients
        optimizer.zero_grad()

        # make minibatch 3d
        minibatch.data.unsqueeze_(1)

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
test_idx = int(len(inputs) * train_frac)
X_test = inputs[test_idx:]
#X_test = X_test[:,:,200:(200+three_secs)]
inputs_tensored = torch.from_numpy(X_test).float()
inputs_tensored.unsqueeze_(1)
outputs = net(Variable(inputs_tensored))
outputs_labels = outputs.max(1)[1].data.numpy().ravel()
print(outputs_labels.shape)

yhat = le.inverse_transform(outputs_labels)
print(yhat)
y_t = np.array(labels[test_idx:])
print(y_t)
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_t, yhat))
print(confusion_matrix(y_t, yhat))
torch.save(net.state_dict(), "output/states/test_model_state")
