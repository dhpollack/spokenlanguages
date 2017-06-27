import numpy as np
from load_data import *
from sklearn.preprocessing import LabelEncoder

#data = np.load("output/sigs_srs_labels.npz")
#sigs, srs, labels = data["sigs"], data["srs"], data["labels"]
#data = None

audio_file_names = load_csv()
sigs, srs, labels = process_audio_files(audio_file_names, N=500)
le = LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)

chromagrams = get_chromagrams(sigs, srs)
mel_spectrograms = get_mel_spectrograms(sigs, srs)

print(chromagrams.shape)
print(mel_spectrograms.shape)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(128*854, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 5)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*854)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

inputs = mel_spectrograms # or chromagrams

b = 4
epochs = 4
train_frac = 0.8
minibatches = int(len(labels)*train_frac / b)
print(minibatches)
print_freq = minibatches // 4
#minibatches = 25

def get_minibatch(X, y, n, b):
    # get the inputs+labels
    mb = X[n:(n+b),:,:]
    l = y[n:(n+b)]
    # wrap them in Variable
    mb, l = Variable(torch.from_numpy(mb).float()), Variable(torch.from_numpy(l))
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
chromagrams_tensored = torch.from_numpy(inputs[-100:]).float()
outputs = net(Variable(chromagrams_tensored))
outputs_labels = outputs.max(1)[1].data.numpy().ravel()
print(outputs_labels.shape)
yhat = le.inverse_transform(outputs_labels)
print(yhat)
y_t = np.array(labels[-100:])
print(y_t)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_t, yhat)
torch.save(net.state_dict(), "models/test_model_state")
