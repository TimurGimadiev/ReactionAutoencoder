from .net import ReactionEncoder
from torch.utils.data import  Subset, DataLoader
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch import no_grad, save, tensor, device, cuda
from numpy import mean
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

cur_device = device("cuda:0" if cuda.is_available() else "cpu")


class NetTrain:
    def __init__(self, data, max_len=100, encoder_hidden=128, decoder_hidden=128,
                 decoder_lstm=256, batch_size=256, save_file="best_model.torch"):
        self.data = data
        self.save_file = save_file
        data.max_len = max_len
        input_size = len(data.tokens)
        self.train = Subset(data,
                       list(set(data.train).intersection([v for k, v in data.sms.items() for v in v if k <= max_len])))
        self.train = DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=10,
                                pin_memory=True, drop_last=False)
        self.test = Subset(data,
                      list(set(data.test).intersection([v for k, v in data.sms.items() for v in v if k <= max_len])))
        self.test = DataLoader(self.test, batch_size=batch_size, shuffle=True, num_workers=10,
                               pin_memory=True, drop_last=False)
        self.crit = CrossEntropyLoss(ignore_index=0)
        self.net = net = ReactionEncoder(input_size, encoder_hidden,
                                         decoder_hidden, decoder_lstm, enc_num_layers=3, dec_num_layers=1).to(cur_device)
        self.opt = Adam(net.parameters(), lr=1e-3)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.opt, mode='max', factor=0.5, patience=1, cooldown=0,
                                                   min_lr=0.0000001, eps=0.00001, verbose=True)
        self.clipping_value = 0.001
        self.best = 0

    def train_epoch(self):
        tr = []
        tq = tqdm(self.train)
        self.net.train()
        for x, t, y in tq:
            self.opt.zero_grad()
            out = self.net((x.to(cur_device), t.to(cur_device)))
            loss = self.crit(out.permute(0, 2, 1),
                        y.to(cur_device))
            cur_tr = loss.item()
            tr.append(cur_tr)
            loss.backward()
            tq.set_postfix({'loss': loss.item()})
            clip_grad_value_(self.net.parameters(), self.clipping_value)
            self.opt.step()
        print(" train_mean: ", mean(tr))
        return mean(tr)

    def validate(self):
        self.net.eval()
        with no_grad():
            val = []
            tq = tqdm(self.train)
            for x, t, y in tq:
                out = self.net((x.to(cur_device), t.to(cur_device)))
                loss = self.crit(out.permute(0, 2, 1), y.to(cur_device))
                cur_val = loss.item()
                val.append(cur_val)
        print(" validation_mean: ", mean(val))
        return mean(val)

    def train_epochs(self, n, val=False, save_best=False):
        best = 0
        for i in range(n):
            score = self.train_epoch()
            if val:
                score = self.validate()
            self.scheduler.step(score)
            if save_best:
                if mean(val) > best:
                    best = mean(val)
                    save(self.net.state_dict(), self.save_file)
                    print("New Best Saved")

    def reconstruct(self):
        self.net.eval()
        with no_grad():
            rec_smi = []
            rec_score = []
            tq = tqdm(self.test)
            for x, t, y in tq:
                out = self.net((x.to(cur_device), t.to(cur_device)))
                pred = self.batch2smi(out, self.data)
                real = self.batch2smi(y, self.data, y=True)
                for p, r in zip(pred, real):
                    rec_score.append(p == r)
                    rec_smi.append((p, r))
            return mean(rec_score), rec_smi

    def batch2smi(self, t, data, y=False):
        if not y:
            pred = t.detach().topk(1).indices.squeeze().tolist()
        else:
            pred = t.detach().tolist()
        stops = []
        for x in pred:
            try:
                stops.append(x.index(data.token2int["STOP"]))
            except ValueError:
                stops.append(0)
        # stops = [x.index(data.token2int["STOP"]) for x in pred]
        striped = [x[:y] for x, y in zip(pred, stops)]
        res = []
        for x in striped:
            res.append("".join(data.int2token[x] for x in x))
        return res

    def generate(self, datagen):
        self.net.eval()
        s = tensor(self.data.smi2class(["START"])[:-1]).to(cur_device)
        dg = DataLoader(datagen, batch_size=self.data.batch.size)
        res = []
        for i in dg:
            w = self.net.decoder.generate(s.repeat(256, 1), i.float().to(cur_device))
            w = w.detach().tolist()
            for z in w:
                try:
                    end = z.index(self.data.token2int["STOP"])
                    res.append("".join(self.data.int2token[x] for x in z[:end]))
                except ValueError:
                    res.append("".join(self.data.int2token[x] for x in z))
        return res

    def latent(self, idx):
        self.net.eval()
        x, t, y = self.data.get_vector(idx)
        res = []
        with no_grad():
            self.net((x.to(cur_device), t.to(cur_device)))
            for i in self.net.neck_out.detach().cpu().numpy():
                res.append(i)
        return res


__all__ = ["NetTrain"]
