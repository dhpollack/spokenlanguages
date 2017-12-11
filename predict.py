import argparse
import torch
import torch.utils.data as data
from loader_voxforge import *
from cfg import CFG

import csv


config = CFG()
"""
data_path_predict = "data/presentation"
vx = VOXFORGE(data_path_predict, langs=config.vx.langs,
              ratios=[0., 0., 1.],
              label_type="lang", use_cache=False,
              use_precompute=False)
vx.transform = config.vx.transform
vx.target_transform = config.vx.target_transform
vx = config.vx
vx.set_split("train")
config.vx = vx
dl = data.DataLoader(vx, batch_size=1, shuffle=False)
"""

config.vx.set_split("test")
RLENC = {v: k for (k, v) in config.vx.target_transform.vocab.items()}

model = config.model
model.eval()
correct = 0

for i, (mb, tgt) in enumerate(config.dl):
    labels = [RLENC[t] for t in tgt]
    if config.use_cuda:
        mb, tgt = mb.cuda(), tgt.cuda()
    mb, tgt = Variable(mb), Variable(tgt)
    out = torch.nn.functional.softmax(model(mb), dim=-1)
    out_print = out.data[0].numpy().tolist()[:3]
    print([(label==RLENC[o], label, RLENC[o], out_print) for o, label in zip(out.data.max(1)[1], labels)])
    correct += (out.data.max(1)[1] == tgt.data).sum()
print("acc: {}".format(correct / len(config.vx)))
