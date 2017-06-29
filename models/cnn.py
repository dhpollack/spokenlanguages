import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_targets):
        super(Net, self).__init__()
        self.features_out_conv = 22848 # dummy value
        self.conv1 = nn.Conv2d(1, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (2, 6))
        self.conv3 = nn.Conv2d(16, 16, (3, 6))
        self.fc1 = nn.Linear(self.features_out_conv, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_targets)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
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
