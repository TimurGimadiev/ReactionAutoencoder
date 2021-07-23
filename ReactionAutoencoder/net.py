import torch
from torch.nn.functional import normalize
from torch.nn import Embedding, Dropout, ReLU, BatchNorm1d, Softmax, LSTM, Linear, Module
from torch.nn.init import orthogonal_


class Encoder(Module):
    def __init__(self, input_size, hidden_size, num_layers=6):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = Embedding(input_size, hidden_size)
        self.drop = Dropout(0.2)
        self.encoder = LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)
        self.relu = ReLU()
        self.norm = BatchNorm1d(hidden_size)
        self.sm = Softmax(-1)

    def forward(self, input):
        bsize, seqsize = input.size()[:2]
        input = self.embed(input)
        input = self.drop(input)
        hid = orthogonal_(torch.empty(self.num_layers * 2, bsize, self.hidden_size))
        cell = orthogonal_(torch.empty(self.num_layers * 2, bsize, self.hidden_size))
        output, (hid, cell) = self.encoder(input, (hid, cell))
        output = cell[-1]
        output = self.norm(output)
        output = normalize(output, p=2, dim=1)
        return output


class Decoder(Module):
    def __init__(self, input_size, hidden_size, lstm_hidden, num_layers=2):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden
        self.embed = Embedding(input_size, 128)
        self.drop = Dropout(0.2)
        self.cell = Linear(hidden_size, lstm_hidden * num_layers)
        self.hid = Linear(hidden_size, lstm_hidden * num_layers)
        self.dec = LSTM(hidden_size, lstm_hidden, batch_first=True, num_layers=num_layers)
        self.final = Linear(lstm_hidden, input_size)
        self.bnorm_input = BatchNorm1d(hidden_size)
        self.bnorm_cell = BatchNorm1d(lstm_hidden)
        self.bnorm_hid = BatchNorm1d(lstm_hidden)
        self.bnorm_dec = BatchNorm1d(lstm_hidden)
        self.bnorm_final = BatchNorm1d(lstm_hidden)
        self.bnorm_norm = BatchNorm1d(lstm_hidden)
        self.relu = ReLU()

    def forward(self, input, out_neck):
        input = self.embed(input)
        input = self.drop(input)
        # cell state generation
        cell = self.cell(out_neck)
        cell = self.relu(cell)
        cell = self.bnorm_cell(cell)
        cell = cell.unsqueeze(0)
        # hid state generation
        hid = self.hid(out_neck)
        hid = self.relu(hid)
        hid = self.bnorm_hid(hid)
        hid = hid.unsqueeze(0)
        # decoding
        output, (hid, _) = self.dec(input, (hid, cell))
        output = output.permute(0, 2, 1)
        output = self.bnorm_final(output)
        output = output.permute(0, 2, 1)
        output = self.relu(self.final(output))
        return output

    def generate(self,smi_t, out_neck):
       max_len = self.data.max_len
       with torch.no_grad():
            # cell state generation
            cell = self.cell(out_neck) # 256, 100, 128
            cell = self.relu(cell)
            cell = self.bnorm_cell(cell)
            cell = cell.unsqueeze(0)
            # hid state generation
            hid = self.hid(out_neck)
            hid = self.relu(hid)
            hid = self.bnorm_hid(hid)
            hid = hid.unsqueeze(0)
            # generation
            for i in range(max_len-1):
                input = self.embed(smi_t)
                output, (hid, _) = self.dec(input, (hid, cell))
                output = output.permute(0, 2, 1)
                output = self.bnorm_final(output)
                output = output.permute(0, 2, 1)
                output = self.relu(self.final(output))
                inds = output.detach().topk(1).indices.squeeze()
                for n, (smi,ind) in enumerate(zip(smi_t, inds)):
                    smi[i+1] = ind[i]
            else:
                for j, (smi,ind) in enumerate(zip(smi_t, inds)):
                    smi_t[j] = torch.cat((smi[1:], ind[max_len-1].unsqueeze(0)), axis=0)
       return smi_t



class ReactionEncoder(Module):
    def __init__(self, input_size, encoder_hidden=128, decoder_hidden=256, decoder_lstm=128,
                 enc_num_layers=3, dec_num_layers=1):
        super(ReactionEncoder, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(input_size, encoder_hidden, num_layers=enc_num_layers)
        self.decoder = Decoder(input_size, decoder_hidden, decoder_lstm, num_layers=dec_num_layers)
        self.relu = ReLU()

    def forward(self, input):
        x, t = input
        self.neck_out = self.encoder(x)
        decoded = self.decoder(t, self.neck_out)
        return decoded


__all__ = ["ReactionEncoder"]
