import argparse
import numpy as np
from load_data import *
from sputils import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models import *

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=8, metavar='b',
                    help='batch size')
parser.add_argument('--freq-bands', type=int, default=224,
                    help='number of frequency bands to use')
parser.add_argument('--window-size', type=int, default=2048,
                    help='size of window for stft')
parser.add_argument('--languages', type=str, nargs='+', default=None,
                    help='languages to filter by')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--use-chromagrams', action='store_true',
                    help='use chromagrams')
parser.add_argument('--log-interval', type=int, default=4,
                    help='reports per epoch')
parser.add_argument('--file-list', type=str, default="data/trainingset.csv",
                    help='csv file with audio files and labels')
parser.add_argument('--grams-path', type=str, default=None,
                    help='path to load spectro/chroma -grams')
parser.add_argument('--model', type=str, default="cnn",
                    help='choose model type')
parser.add_argument('--load-model', action='store_true',
                    help='load model from disk')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
args = parser.parse_args()

def transform(tensor, model = "cnn"):
    if model == "cnn":
        return(Variable(tensor.unsqueeze(1)))
    elif model == "resnet":
        return(Variable(torch.stack([tensor, tensor, tensor], dim=1)))

def lr_scheduler(optimizer, epoch, step_size=7, gamma=0.1, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (gamma**(epoch // step_size))

    if epoch % step_size == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
# set seed
torch.manual_seed(args.seed)

# load spectro/chroma -grams
inputs, labels = get_grams(use_chromagrams = args.use_chromagrams, N = None,
                           grams_path = args.grams_path, languages = args.languages,
                           window_size = args.window_size, freq_bands = args.freq_bands)
le = LabelEncoder()
le.fit(spconfig.lang_classes)
num_targets = le.classes_.shape[0]
labels_encoded = le.transform(labels)


# Create Network
gtype = "chromagrams" if args.use_chromagrams else "spectrograms"
if args.model == "cnn":
    nn_builder = cnn.Net
    nnargs = {"input_dim": (1,) + inputs.shape[1:]}
elif args.model == "resnet":
    nn_builder = resnet.resnetX
    nnargs = {}
else:
    print("unknown model type")
model_save_path = "output/states/"+args.model+"_model_"+gtype+".pt"
#net = ResNet(BasicBlock, [2, 2, 0, 2], num_classes=num_targets)
net = nn_builder(num_classes=num_targets, **nnargs)
if args.load_model:
    net.load_state_dict(torch.load(model_save_path))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
print(net)

# create training and testing sets
b = args.batch_size
shuffle_inputs = True
if args.validate:
    train_frac = 0.8
else:
    train_frac = 1.0
train_idx = int(len(inputs) * train_frac)
inputs_training = torch.from_numpy(inputs[:train_idx]).float()
labels_training = labels_encoded[:train_idx]

labels_training = torch.from_numpy(labels_training)
trainset = TensorDataset(inputs_training, labels_training)
trainloader = DataLoader(trainset, batch_size=b, shuffle=shuffle_inputs)

print("Batch Size:", b)
if args.validate:
    inputs_testing = torch.from_numpy(inputs[train_idx:]).float()
    labels_testing = labels_encoded[train_idx:]

# calculate number of minibatches
# per epoch variables
epochs = args.epochs
minibatches = train_idx // b
print_freq = minibatches // args.log_interval
print("Epochs:", epochs)
print("Minibatches Per Epoch:", minibatches)
print("Print Frequency (minibatches):", print_freq)
net.train() # set model into training mode
for epoch in range(epochs):  # loop over the dataset multiple times
    # reset vars each epoch
    optimizer = lr_scheduler(optimizer, epoch, init_lr=args.lr)
    running_loss = 0.0
    n = 0
    for i, (minibatch, l) in enumerate(trainloader):
        # minibatch to variable
        minibatch, l = transform(minibatch, args.model), Variable(l)
        n += l.size(0)
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
                  (epoch + 1, i + 1, (i*b), running_loss / n)) # average loss in epoch
    if args.validate:
        net.eval() # set model into evaluation mode
        output = []
        for t in torch.split(inputs_testing, args.batch_size):
            output += [net(transform(t, args.model)).data]
        output = torch.cat(output)
        outputs_labels = output.max(1)[1].numpy().ravel()
        print("Validation Accuracy: %.2f"%accuracy_score(labels_testing, outputs_labels))
        net.train() # reset to training mode
print('Finished Training')

# Final Validation Prediction
if args.validate:
    net.eval() # set model into evaluation mode
    yhat = le.inverse_transform(outputs_labels)
    y_t = le.inverse_transform(labels_testing)
    print(sorted([x for x in zip(yhat,y_t)]))
    print(accuracy_score(y_t, yhat))
    print(confusion_matrix(y_t, yhat))

# save model parameters
if args.save_model:
    torch.save(net.state_dict(), model_save_path)
