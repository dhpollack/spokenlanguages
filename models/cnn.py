import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

conv_layers = [((1, 4, 7, 1), (2, 2)),
               ((4, 12, (2,6),(1,2)), (2, 2)),
               ((12, 24, (3, 6), (2, 2)), None)]

class Net(nn.Module):
    def __init__(self, num_classes, input_dim = (1, 224, 427)):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, (2, 6), (1,2))
        self.conv3 = nn.Conv2d(32, 64, (3, 6), (1,1))
        self.fc1 = nn.Linear(self._calc_conv_out(input_dim), 1024)
        #self.fc1_calc = self._make_layer(self.num_flat_features(x), 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _calc_conv_out(self, v):
        v = self._pool_dim(self._conv_dim(v, (7, 7), (1, 1), (0, 0), 16), (2, 2))
        v = self._pool_dim(self._conv_dim(v, (2, 6), (1, 2), (0, 0), 32), (2, 2))
        v = self._pool_dim(self._conv_dim(v, (3, 6), (1, 1), (0, 0), 64), (2, 2))
        return(reduce(lambda x, y: x*y, v))

    def _conv_dim(self, vol, f, s, p, k):
        return((k, (vol[1]-f[0]+2*p[0])//s[0] + 1, (vol[2]-f[1]+2*p[1])//s[1] + 1))

    def _pool_dim(self, vol, s):
        return((vol[0], vol[1]//s[0], vol[2]//s[1]))
