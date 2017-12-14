import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, input, hidden):
        output = input
        output, hidden = self.gru(output, hidden)
        #print("encoder:", output.size(), hidden.size())
        return output, hidden

    def initHidden(self, ttype=None):
        if ttype == None:
            ttype = torch.FloatTensor
        result = Variable(ttype(self.n_layers * 1, self.batch_size, self.hidden_size).fill_(0))
        return result.cuda() if torch.cuda.is_available() else result

class Attn(nn.Module):
    def __init__(self, hidden_size, batch_size=1, method="dot"):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(batch_size, 1, hidden_size))

    def forward(self, hidden, encoder_outputs):

        # get attn energies in one batch
        attn_energies = self.score(hidden, encoder_outputs)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, dim=2)

    def score(self, hidden, encoder_output):
        #print("attn.score:", hidden.size(), encoder_output.size())
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.transpose(2, 1)
            energy = hidden.bmm(energy)
            return energy

        elif self.method == 'concat':
            hidden = hidden * Variable(encoder_output.data.new(encoder_output.size()).fill_(1)) # broadcast hidden to encoder_outputs size
            energy = self.attn(torch.cat((hidden, encoder_output), -1))
            energy = energy.transpose(2, 1)
            energy = self.v.bmm(energy)
            return energy
        else:
            #self.method == 'dot':
            encoder_output = encoder_output.transpose(2, 1)
            energy = hidden.bmm(encoder_output)
            return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_model="dot", n_layers=1, dropout=0.1, batch_size=1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size

        # Define layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(hidden_size, method=attn_model, batch_size=batch_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: This now runs in batch but was originally run one
        #       step at a time
        #       B = batch size
        #       S = output length
        #       N = # of hidden features

        # Get the embedding of the current input word (last output word)
        batch_size = self.batch_size # without PackedSequence calc as input_seq.size(0)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(input_seq, last_hidden)
        encoder_outputs, encoder_lengths = unpack(encoder_outputs, batch_first=True)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs) # [B, S, L] dot [B, L, N] -> [B, S, N]
        #print(attn_weights.size(), encoder_outputs.size(), context.size())
        #print("decoder context:", context.size())

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        concat_input = torch.cat((rnn_output, context), -1) # B x S x 2*N
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

def attn(kwargs_encoder, kwargs_decoder):
    encoder = EncoderRNN(**kwargs_encoder)
    decoder = LuongAttnDecoderRNN(**kwargs_decoder)

    return [encoder, decoder]
