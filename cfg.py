import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
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
        elif "resnet101" in self.model_name:
            model = models.resnet.resnet101(use_pretrained, num_langs=5)
            if not use_pretrained:
                model.load_state_dict(torch.load(args.load_model, map_location=lambda storage, loc: storage))
        elif "attn" in self.model_name:
            self.hidden_size = 500
            kwargs_encoder = {
                "input_size": args.freq_bands,
                "hidden_size": self.hidden_size,
                "n_layers": 1,
                "batch_size": args.batch_size
            }
            kwargs_decoder = {
                "hidden_size": self.hidden_size,
                "output_size": 5,
                "attn_model": "general",
                "n_layers": 1,
                "dropout": 0.0, # was 0.1
                "batch_size": args.batch_size
            }
            model = models.attn.attn(kwargs_encoder, kwargs_decoder)
        if self.use_cuda:
            if isinstance(model, list):
                model = [m.cuda() for m in model]
            else:
                model = model.cuda()
            ngpu = torch.cuda.device_count()
            if ngpu > 1:
                print("Detected {} CUDA devices")
                if isinstance(model, list):
                    model = [torch.nn.DataParallel(m) for m in model]
                else:
                    model = torch.nn.DataParallel(model)
        return model

    def get_dataloader(self):
        vx = VOXFORGE(args.data_path, langs=args.languages,
                      label_type="lang", use_cache=args.use_cache,
                      use_precompute=args.use_precompute)
        if self.model_name == "resnet34_conv" or self.model_name == "resnet101_conv":
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
        elif "attn" in self.model_name:
            T = tat.Compose([
                    tat.MEL(n_mels=224),
                    spl_transforms.SqueezeDim(2),
                    tat.LC2CL(),
                    #tat.BLC2CBL(),
                ])
            TT = spl_transforms.LENC(vx.LABELS)
        vx.transform = T
        vx.target_transform = TT
        if args.use_precompute:
            vx.load_precompute(args.model_name)
        dl = data.DataLoader(vx, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)
        if "attn" in self.model_name:
            dl.collate_fn = pad_packed_collate
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
        elif "resnet101" in self.model_name:
            self.epochs = [("fc_layer", 20), ("full_model", 50)]
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
        elif "attn" in self.model_name:
            self.epochs = [("full_model", 100)]
            self.criterion = nn.CrossEntropyLoss()
            self.L["full_model"] = {}
            self.L["full_model"]["optimizer"] = torch.optim.RMSprop
            self.L["full_model"]["params"] = [
                    {"params": self.model[0].parameters()},
                    {"params": self.model[1].parameters()}
                ]
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
        if "resnet" in self.model_name:
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
        if "attn" in self.model_name:
            self.vx.set_split("train")
            self.optimizer = self.get_optimizer(epoch)
            epoch_losses = []
            encoder = self.model[0]
            decoder = self.model[1]
            input_type = torch.FloatTensor
            for i, ((mb, lengths), tgts) in enumerate(self.dl):
                # set model into train mode and clear gradients
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()

                # set inputs and targets
                #if i == 0: print(mb.size())
                #mb = mb.transpose(2, 1) # [B x N x L] -> [B, L, N]
                if isinstance(mb, torch.nn.utils.rnn.PackedSequence):
                    pass
                else:
                    mb = Variable(mb)
                tgts = Variable(tgts)
                #print(mb.size(), tgts.size())
                encoder_hidden = encoder.initHidden(input_type)
                encoder_output, encoder_hidden = encoder(mb, encoder_hidden)

                # Prepare input and output variables for decoder
                dec_size = [[[0] * encoder.hidden_size]*1]*args.batch_size
                #print(encoder_output.data.new(dec_size).size())
                enc_out_var, enc_out_len = unpack(encoder_output, batch_first=True)
                dec_i = Variable(enc_out_var.data.new(dec_size))
                #dec_i = Variable(encoder_output.data.new(dec_size))
                #dec_i = encoder_output
                dec_h = encoder_hidden # Use last (forward) hidden state from encoder
                #print(decoder.n_layers, encoder_hidden.size(), dec_i.size(), dec_h.size())

                """
                # Run through decoder one time step at a time
                # collect attentions
                attentions = []
                outputs = []
                dec_i = Variable(torch.FloatTensor([[[0] * hidden_size] * batch_size]))
                target_seq = Variable(torch.FloatTensor([[[-1] * hidden_size]*output_length]))
                for t in range(output_length):
                    #print("t:", t, dec_i.size())
                    dec_o, dec_h, dec_attn = decoder2(
                        dec_i, dec_h, encoder2_output
                    )
                    #print("decoder output", dec_o.size())
                    dec_i = target_seq[:,t].unsqueeze(1) # Next input is current target
                    outputs += [dec_o]
                    attentions += [dec_attn]
                dec_o = torch.cat(outputs, 1)
                dec_attn = torch.cat(attentions, 1)
                """
                # run through decoder in one shot
                dec_o, dec_h, dec_attn = decoder(dec_i, dec_h, encoder_output)
                #print(dec_o)
                print(dec_o.size(), dec_h.size(), dec_attn.size())
                #print(dec_o.view(-1, decoder.output_size).size(), tgts.view(-1).size())

                # calculate loss and backprop
                loss = self.criterion(dec_o.view(-1, decoder.output_size), tgts.view(-1))
                #nn.utils.clip_grad_norm(encoder.parameters(), 0.05)
                #nn.utils.clip_grad_norm(decoder.parameters(), 0.05)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])
                print(loss.data[0])
                if i % args.log_interval == 0 and args.validate and i != 0:
                    self.validate(epoch)
                self.vx.set_split("train")
                self.train_losses.append(epoch_losses)


    def validate(self, epoch):
        if "resnet" in self.model_name:
            self.model.eval()
            self.vx.set_split("valid")
            running_validation_loss = 0
            correct = 0
            num_batches = len(self.dl)
            for mb_valid, tgts_valid in self.dl:
                if self.use_cuda:
                    mb_valid, tgts_valid = mb_valid.cuda(), tgts_valid.cuda()
                mb_valid, tgts_valid = Variable(mb_valid), Variable(tgts_valid)
                out_valid = self.model(mb_valid)
                loss_valid = self.criterion(out_valid, tgts_valid)
                running_validation_loss += loss_valid.data[0]
                correct += (out_valid.data.max(1)[1] == tgts_valid.data).sum()
            self.valid_losses.append((running_validation_loss / num_batches, correct / len(self.vx)))
            print("loss: {}, acc: {}".format(running_validation_loss / num_batches, correct / len(self.vx)))

    def get_train(self):
        return self.fit

    def save(self, epoch):
        mstate = self.model.state_dict()
        torch.save(mstate, "output/states/{}_{}.pt".format(self.model_name, epoch+1))

    def precompute(self, m):
        if "resnet" in self.model_name:
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

def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.

       Args:
         batch: (list of tuples) [(audio, target)].
             audio is a FloatTensor
             target is a LongTensor with a length of 8
       Output:
         packed_batch: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         labels: (Tensor), labels from the file names of the wav.

    """
    use_cuda = torch.cuda.is_available()
    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        lengths = [sigs.size(0)]
        #sigs = sigs.t()
        sigs.unsqueeze_(0)
        labels = tensor.LongTensor([labels]).unsqueeze(0)
    if len(batch) > 1:
        sigs, labels, lengths = zip(*[(a, b, a.size(0)) for (a,b) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
        max_len, n_feats = sigs[0].size()
        sigs = [pad_sig(s, max_len, n_feats) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        labels = torch.LongTensor(labels).unsqueeze(0)
    if use_cuda:
        sigs, labels = sigs.cuda(), labels.cuda()
    packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    return (packed_batch, lengths), labels

def pad_sig(s, max_len, n_feats):
    s_len = s.size(0)
    s_new = s.new(max_len, n_feats).fill_(0)
    s_new[:s_len] = s
    return s_new
