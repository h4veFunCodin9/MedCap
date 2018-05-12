import torch
import torch.nn.functional as F

class SentDecoder(torch.nn.Module):
    def __init__(self, config):
        super(SentDecoder, self).__init__()
        self.hidden_size = config.SentLSTM_HiddenSize
        self.topic_size = config.TopicSize
        # context vector for each time step
        self.ctx_im_W = torch.nn.Linear(config.IM_EmbeddingSize, self.hidden_size)
        self.ctx_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # RNN unit
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        # topic output
        self.topic_h_W = torch.nn.Linear(self.hidden_size, self.topic_size)
        self.topic_ctx_W = torch.nn.Linear(self.hidden_size, self.topic_size)
        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_W = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, x, hidden):
        # generate current context vector
        ctx = self.ctx_im_W(x) + self.ctx_h_W(hidden)
        ctx = F.tanh(ctx)
        # run RNN
        prev_hidden = hidden
        output, hidden = self.gru(ctx, hidden)
        output = output[0]
        # predict topic vector
        topic = self.topic_h_W(output) + self.topic_ctx_W(ctx)
        topic = F.tanh(topic)
        # predict stop distribution
        stop = self.stop_h_W(output) + self.stop_prev_h_W(prev_hidden)
        stop = F.tanh(stop)
        stop = self.stop_W(stop)

        return topic, stop, hidden

    def init_hidden(self):
        hidden = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size))
        hidden = hidden.cuda() if torch.cuda.is_available() else hidden
        return hidden


class WordDecoder(torch.nn.Module):
    def __init__(self, config):
        super(WordDecoder, self).__init__()
        self.hidden_size = config.WordLSTM_HiddenSize

        self.embedding = torch.nn.Embedding(config.DICT_SIZE, self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, config.DICT_SIZE)

    def forward(self, x, hidden):
        #print('input:', input)
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden