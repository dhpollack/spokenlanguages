import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import models
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import spl_transforms
from loader_voxforge import *
import math

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
parser.add_argument('--use-precompute', action='store_true',
                    help='precompute transformations')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for data loader')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--chkpt-interval', type=int, default=10,
                    help='how often to save checkpoints')
parser.add_argument('--model-name', type=str, default="resnet34_conv",
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


class CFG(object):
    def __init__(self):
        self.max_len = 150000
        self.use_cuda = torch.cuda.is_available()
        print("CUDA: {}".format(self.use_cuda))
        self.model_name = args.model_name
        self.model = self.get_model()
        self.vx, self.dl = self.get_dataloader()
        self.criterion, self.optimizer = self.init_optimizer()
        self.save_model = args.save_model
        self.chkpt_interval = args.chkpt_interval
        self.valid_losses = []
        self.train_losses = []

    def get_model(self):
        use_pretrained = True if args.load_model is None else False
        if "resnet34" in self.model_name:
            model = models.resnet.resnet34(use_pretrained, num_langs=5)
            if not use_pretrained:
                model.load_state_dict(torch.load(args.load_model, map_location=lambda storage, loc: storage))
        model = model.cuda() if self.use_cuda else model
        return model

    def get_dataloader(self):
        vx = VOXFORGE(args.data_path, langs=args.languages,
                      label_type="lang", use_cache=args.use_cache,
                      use_precompute=args.use_precompute)
        if self.model_name == "resnet34_conv":
            T = tat.Compose([
                    #tat.PadTrim(self.max_len),
                    tat.MEL(n_mels=224),
                    tat.BLC2CBL(),
                    tvt.ToPILImage(),
                    tvt.Resize((224, 224)),
                    tvt.ToTensor(),
                ])
            TT = spl_transforms.LENC(vx.LABELS)
        elif self.model_name == "resnet34_mfcc":
            sr = 16000
            ws = 800
            hs = ws // 2
            n_fft = 512 # 256
            n_filterbanks = 26
            n_coefficients = 12
            low_mel_freq = 0
            high_freq_mel = (2595 * math.log10(1 + (sr/2) / 700))
            mel_pts = torch.linspace(low_mel_freq, high_freq_mel, n_filterbanks + 2) # sr = 16000
            hz_pts = torch.floor(700 * (torch.pow(10,mel_pts / 2595) - 1))
            bins = torch.floor((n_fft + 1) * hz_pts / sr)
            td = {
                    "RfftPow": spl_transforms.RfftPow(n_fft),
                    "FilterBanks": spl_transforms.FilterBanks(n_filterbanks, bins),
                    "MFCC": spl_transforms.MFCC(n_filterbanks, n_coefficients),
                 }

            T = tat.Compose([
                    tat.Scale(),
                    #tat.PadTrim(self.max_len, fill_value=1e-8),
                    spl_transforms.Preemphasis(),
                    spl_transforms.Sig2Features(ws, hs, td),
                    spl_transforms.DummyDim(),
                    tat.BLC2CBL(),
                    tvt.ToPILImage(),
                    tvt.Resize((224, 224)),
                    tvt.ToTensor(),
                ])
            TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        if args.use_precompute:
            vx.load_precompute(args.model_name)
        dl = data.DataLoader(vx, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)
        return vx, dl

    def init_optimizer(self):
        self.L = {}
        if "resnet34" in self.model_name:
            self.epochs = [("fc_layer", 40), ("full_model", 100)]
            self.criterion = nn.CrossEntropyLoss()
            self.L["fc_layer"] = {}
            self.L["fc_layer"]["optimizer"] = torch.optim.Adam
            self.L["fc_layer"]["params"] = self.model[1].fc.parameters()
            self.L["fc_layer"]["optim_kwargs"] = {"lr": 0.0001,}
            self.L["fc_layer"]["precompute"] = nn.Sequential(self.model[0], *list(self.model[1].children())[:-1])
            self.L["fc_layer"]["model"] = self.model[1].fc
            self.L["full_model"] = {}
            self.L["full_model"]["optimizer"] = torch.optim.SGD
            self.L["full_model"]["params"] = self.model.parameters()
            self.L["full_model"]["optim_kwargs"] = {"lr": 0.0001, "momentum": 0.9,}
            self.L["full_model"]["model"] = self.model
        optim_layer, _ = self.epochs[0]
        opt = self.L[optim_layer]["optimizer"]
        params = self.L[optim_layer]["params"]
        kwargs = self.L[optim_layer]["optim_kwargs"]
        print("Initializing optimizer: {} with args {}".format(opt, kwargs))
        return self.criterion, opt(params, **kwargs)

    def get_optimizer(self, epoch):
        grenz = 0
        for i, (k, v) in enumerate(self.epochs):
            grenz += v
            if epoch == 0:
                """optimizer alread initialized
                print(k, epoch, grenz)
                opt = self.L[k]["optimizer"]
                params = self.L[k]["params"]
                kwargs = self.L[k]["optim_kwargs"]
                print("Using new optimizer: {} with args {}".format(opt, kwargs))
                return create_optimizer(opt, params, kwargs)
                """
                pass
            elif epoch == grenz:
                k_next, _ = self.epochs[i+1]
                print(k_next, epoch, grenz)
                opt = self.L[k_next]["optimizer"]
                params = self.L[k_next]["params"]
                kwargs = self.L[k_next]["optim_kwargs"]
                print("Using new optimizer: {} with args {}".format(opt, kwargs))
                return opt(params, **kwargs)
            else:
                pass
        return self.optimizer

    def fit(self, epoch):
        if "resnet34" in self.model_name:
            if args.use_precompute:
                pass # TODO implement network precomputation
                #self.precompute(self.L["fc_layer"]["precompute"])
            self.vx.set_split("train")
            self.optimizer = self.get_optimizer(epoch)
            epoch_losses = []
            for i, (mb, tgts) in enumerate(self.dl):
                self.model.train()
                if self.use_cuda:
                    mb, tgts = mb.cuda(), tgts.cuda()
                mb, tgts = Variable(mb), Variable(tgts)
                self.model.zero_grad()
                out = self.model(mb)
                loss = self.criterion(out, tgts)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])
                print(loss.data[0])
                if i % args.log_interval == 0 and args.validate and i != 0:
                    self.validate(epoch)
                self.vx.set_split("train")
            self.train_losses.append(epoch_losses)

    def validate(self, epoch):
        if "resnet34" in self.model_name:
            self.model.eval()
            self.vx.set_split("valid")
            running_validation_loss = 0
            correct = 0
            for mb_valid, tgts_valid in self.dl:
                if self.use_cuda:
                    mb_valid, tgts_valid = mb_valid.cuda(), tgts_valid.cuda()
                mb_valid, tgts_valid = Variable(mb_valid), Variable(tgts_valid)
                out_valid = self.model(mb_valid)
                loss_valid = self.criterion(out_valid, tgts_valid)
                running_validation_loss += loss_valid.data[0]
                correct += (out_valid.data.max(1)[1] == tgts_valid.data).sum()
            self.valid_losses.append((running_validation_loss, correct / len(self.vx)))
            print("loss: {}, acc: {}".format(running_validation_loss, correct / len(self.vx)))

    def get_train(self):
        return self.fit

    def save(self, epoch):
        mstate = self.model.state_dict()
        torch.save(mstate, "output/states/{}_{}.pt".format(self.model_name, epoch+1))

    def precompute(self, m):
        if "resnet34" in self.model_name:
            dl = data.DataLoader(self.vx, batch_size=args.batch_size,
                                 num_workers=args.num_workers, shuffle=False)
            m.eval()
            for splt in ["train", "valid"]:
                self.vx.set_split(splt)
                c = self.vx.splits[splt].start
                for i, (mb, tgts) in enumerate(dl):
                    bs = mb.size(0)
                    if self.use_cuda:
                        mb = mb.cuda()
                    mb = Variable(mb)
                    m.zero_grad()
                    out = m(mb).data.cpu()
                    for j_i, j_k in enumerate(range(c, c+bs)):
                        idx_split = self.vx.splits[splt][j_k]
                        k = self.vx.data[idx_split]
                        self.vx.cache[k] = (out[j_i], tgts[j_i])
                    c += bs
