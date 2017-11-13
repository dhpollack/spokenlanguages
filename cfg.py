import argparse
import torch
import torch.nn as nn
import models
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import spl_transforms

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=100, metavar='b',
                    help='batch size')
parser.add_argument('--freq-bands', type=int, default=224,
                    help='number of frequency bands to use')
parser.add_argument('--languages', type=str, nargs='+', default=["de", "en", "es"],
                    help='languages to filter by')
parser.add_argument('--data-path', type=str, default="data/voxforge",
                    help='data path')
parser.add_argument('--use-cache', action='store_true',
                    help='use cache in the dataloader')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for data loader')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--chkpt-interval', type=int, default=10,
                    help='how often to save checkpoints')
parser.add_argument('--model-name', type=str, default="resnet34",
                    help='data path')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
parser.add_argument('--train-full-model', action='store_true',
                    help='train full model vs. final layer')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

MODELS = {
    "resnet34": {
        "T": tat.Compose([
                tat.PadTrim(150000),
                tat.MEL(n_mels=224),
                tat.BLC2CBL(),
                tvt.ToPILImage(),
                tvt.Scale((224, 224)),
                tvt.ToTensor(),
             ]),
         "model": models.resnet.resnet34(True, num_langs=5),
         "state_dict_path": args.load_model,
    }
}

TRAINING = {
    "resnet34": {
        "epochs": [("fc_layer", 40), ("full_model", 100)],
        "criterion": nn.CrossEntropyLoss(),
        "fc_layer": {
            "optimizer": torch.optim.Adam,
            "params": MODELS["resnet34"]["model"][1].fc.parameters(),
            "optim_kwargs": {"lr": 0.0001,},
        },
        "model": {
            "optimizer": torch.optim.SGD,
            "params": MODELS["resnet34"]["model"].parameters(),
            "optim_kwargs": {"lr": 0.0001, "momentum": 0.9,},
        }
    }
}
