import argparse
import torch
from cfg import CFG

import csv

config = CFG()
config.vx.precompute_transforms()
torch.save(config.vx.cache, "output/features/{}_features.pt".format(config.model_name))
