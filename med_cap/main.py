import torch
import torchvision.models as M
import torch.nn.functional as F

from PIL import Image
import numpy as np
import unicodedata
import re, random
import time, math
import os, pickle

import skimage.transform as T

from config import Config

import matplotlib
matplotlib.use('agg') # https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab/4706614#4706614
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

######################
# Prepare Dataset
######################

SOS_INDEX = 0
EOS_INDEX = 1


class Lang:
    '''
    The dictionary for a language
    '''
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = {0:'SOS', 1:'EOS'}
        self.word2count = {}
        self.n_words = 2

    def addSentence(self, s, mode='word'):
        if mode == 'char':
            terms = list(s)
        elif mode == 'word':
            import fool
            terms = fool.cut(s)
        else:
            print("Unknow mode {}.".format(mode))
            return
        terms = terms[0]
        for w in terms:
            w = w.strip()
            if len(w) > 0:
                self.addWord(w)

    def addWord(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = self.n_words
            self.idx2word[self.n_words] = w
            self.word2count[w] = 1
            self.n_words += 1
        else:
            self.word2count[w] += 1

    def __len__(self):
        return self.n_words


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([。*])", r" \1 ", s)   #将标点符号用空格分开
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  #除字母标点符号的其他连续字符替换成一个空格
    return s

def read_captions(annFile, image_root):
    dataset = open(annFile, 'r').read()
    dataset = dataset.split('\n')
    pairs = []
    for p_str in dataset:
        if len(p_str) <= 1:
            continue
        image_id, caption, summary = p_str.split('\t')
        caption = normalize_string(caption)
        caption = [sent.strip() for sent in caption.split(' 。 ') if len(sent.strip()) > 0]
        pairs.append((os.path.join(image_root, image_id+'.npy'), caption))
    return pairs

def split_train_val(pairs, val_prop=0.1):
    # return train pairs and val pairs
    num = len(pairs)
    import random
    random.shuffle(pairs)
    return pairs[int(num*val_prop):], pairs[:int(num*val_prop)]


def prepare_dict(pairs, mode='word'):
    lang = Lang("ch")
    for pair in pairs:
        for sent in pair[1]:
            lang.addSentence(sent, mode=mode)
    print("Language Dictionary: ", lang.n_words)
    return lang


def stat_lang(lang):
    stat = lang.word2count.copy()
    stat = [(k, stat[k]) for k in stat.keys()]
    stat.sort(key=lambda x: x[1], reverse=True)
    total = np.sum([s[1] for s in stat])
    cnt = 0
    num = 0
    for item in stat:
        num += 1
        cnt += item[1]
        if cnt > total * 0.995:
            break
    print(num, 'words take up 99.5% occurrence.')

# find max number of sentences; find max length of sentences
def stat_captions(pairs, mode='word'):
    w_max, s_max = 0, 0
    for path, caption in pairs:
        s_max = s_max if len(caption) < s_max else len(caption)
        for sent in caption:
            if mode == 'word':
                import fool
                cur_len = len([term.strip() for term in fool.cut(sent)[0] if len(term.strip())])
            elif mode == 'char':
                cur_len = len([w.strip() for w in sent if len(w.strip())>0])
            else:
                print("Unknow mode.")
                return
            w_max = w_max if cur_len < w_max else cur_len
    print("Max number of sentences: ", s_max)
    print("Max length of sentences: ", w_max)

def variable_from_image_path(image_path, load_fn):
    im = np.array(load_fn(image_path))
    #im = T.resize(im, (512, 512, 3), mode='reflect')

    im_data = np.zeros([1, 4, 240, 240])  # IU chest X-Ray (COCO 640x480)
    im_data[0, 0, ...], im_data[0, 1, ...], im_data[0, 2, ...], im_data[0, 3, ...] = im[0, :, :, 75], im[1, :, :, 75], im[2, :, :, 75], im[3, :, :, 75]
    seg_data = np.zeros([1, 240, 240])
    seg_data[0, ...] = im[4, :, :, 75]
    seg_data[seg_data==4] = 3
    seg_data = seg_data.astype(np.int16)

    im_data = torch.autograd.Variable(torch.FloatTensor(im_data))
    im_data = im_data.cuda() if torch.cuda.is_available() else im_data

    seg_data = torch.autograd.Variable(torch.LongTensor(seg_data))
    seg_data = seg_data.cuda() if torch.cuda.is_available() else seg_data
    return im_data, seg_data


def variable_from_caption(lang, cap, max_sent_num, mode='word'):
    indices = []
    for sent in cap:
        if mode == 'word':
            import fool
            terms = fool.cut(sent)
        elif mode == 'char':
            terms = list(sent)
        else:
            print('Unknown mode...')
            return None
        terms = terms[0]
        indices.append([lang.word2idx[term.strip()] for term in terms if len(term.strip())>0])
    stop = [0 if i < len(indices) else 1 for i in range(max_sent_num)]

    max_len = max([len(sent) for sent in indices])
    # append End_Of_Sequence token; increase to the same size
    for sent in indices:
        sent.extend([EOS_INDEX] * (max_len - len(sent) + 1))

    indices = torch.autograd.Variable(torch.LongTensor(indices)).view(-1, max_len + 1)
    stop = torch.autograd.Variable(torch.LongTensor(stop)).view(-1, 1)
    return indices.cuda() if torch.cuda.is_available() else indices, stop.cuda() if torch.cuda.is_available() else stop

def variables_from_pair(lang, pair, max_sent_num, im_load_fn):
    cap_var, stop_var = variable_from_caption(lang, pair[1], max_sent_num)
    image_var, seg_var = variable_from_image_path(pair[0], load_fn=im_load_fn)
    return image_var, seg_var, cap_var, stop_var


#############################
# Model: Encoder and Decoder
##############################
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding_size = config.IM_EmbeddingSize

        # contracting path
        self.conv1 = ConvBlock(4, 64)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBlock(256, 512)
        self.mp4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(512, 1024)

        # expanding path
        self.up1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv1 = ConvBlock(1024, 512)
        self.up2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv2 = ConvBlock(512, 256)
        self.up3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv3 = ConvBlock(256, 128)
        self.up4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.outconv = torch.nn.Conv2d(128, config.SegClasses, kernel_size=1, padding=0)

        # image embedding: deep layers
        shape = config.FeatureShape
        self.linear = torch.nn.Linear(in_features=(shape[0] * shape[1] * shape[2]), out_features=self.embedding_size)

    def forward(self, x):   # BRATS image: [1, 4, 240, 240]
        # contracting
        x_240 = self.conv1(x)
        x_120 = self.mp1(x_240)
        x_120 = self.conv2(x_120)
        x_60 = self.mp2(x_120)
        x_60 = self.conv3(x_60)
        x_30 = self.mp3(x_60)
        x_30 = self.conv4(x_30)
        x_15 = self.mp4(x_30)
        x_15 = self.conv5(x_15)

        # expanding
        _x_30 = torch.cat([self.up1(x_15), x_30], dim=1)
        _x_30 = self.deconv1(_x_30)
        _x_60 = torch.cat([self.up2(_x_30), x_60], dim=1)
        _x_60 = self.deconv2(_x_60)
        _x_120 = torch.cat([self.up3(_x_60), x_120], dim=1)
        _x_120 = self.deconv3(_x_120)
        _x_240 = torch.cat([self.up4(_x_120), x_240], dim=1)
        output = self.outconv(_x_240)

        feature = x_15
        embedding = self.linear(feature.view(-1))
        return embedding.view(1, -1), output


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


def save_model(encoder, sent_decoder, word_decoder, config, suffix=""):
    # save the model
    print("Saving models...")
    torch.save(encoder.state_dict(), os.path.join(config.StoreRoot, "encoder_"+suffix))
    if not config.OnlySeg:
        torch.save(sent_decoder.state_dict(), os.path.join(config.StoreRoot, "sent_decoder_"+suffix))
        torch.save(word_decoder.state_dict(), os.path.join(config.StoreRoot, "word_decoder_"+suffix))
    print("Done!")


def load_model(encoder, sent_decoder, word_decoder, load_root):
    encoder_path = os.path.join(load_root, 'encoder')
    sent_decoder_path = os.path.join(load_root, 'sent_decoder')
    word_decoder_path = os.path.join(load_root, 'word_decoder')

    if os.path.isfile(encoder_path):
        print("Loading the model for 'encoder' from '{}' ...".format(encoder_path))
        encoder.load_state_dict(torch.load(encoder_path))
        print("Loaded!")

    if os.path.isfile(sent_decoder_path):
        print("Loading the model for 'sent_decoder' from '{}' ...".format(sent_decoder_path))
        sent_decoder.load_state_dict(torch.load(sent_decoder_path))
        print("Loaded!")

    if os.path.isfile(word_decoder_path):
        print("Loading the model for 'word_decoder' from '{}' ...".format(word_decoder_path))
        word_decoder.load_state_dict(torch.load(word_decoder_path))
        print("Loaded!")

###################################
# Train and evaluate functionality
###################################
def train(input_variables, seg_target_variables, cap_target_variables, stop_target_variables, encoder, sent_decoder, word_decoder,
          encoder_optimizer, sent_decoder_optimizer, word_decoder_optimizer, criterion, config):

    encoder_optimizer.zero_grad()
    if not config.OnlySeg:
        sent_decoder_optimizer.zero_grad()
        word_decoder_optimizer.zero_grad()

    seg_loss, stop_loss, cap_loss = 0, 0, 0

    total_sent_num, total_word_num = 0, 0

    # input_variable: 1 x 3 x 512 x 512, cap_target_variable: num_sent x len_sent, stop_target_variable: max_num_sent x 1
    def _one_pass(input_variable, seg_target_variable, cap_target_variable, stop_target_variable, only_seg):

        _im_embedding, pred_seg = encoder(input_variable) # [1, HiddenSize]

        # compute segmentation loss
        _seg_loss = criterion(pred_seg, seg_target_variable)

        if only_seg:
            return _seg_loss, 0, 0, 0, 0

        _sent_num = cap_target_variable.size()[0]
        _word_num = 0

        _cap_loss = 0
        _stop_loss = 0

        # sentence LSTM
        sent_decoder_hidden = sent_decoder.init_hidden()
        sent_decoder_input = _im_embedding  # not changed

        # generate topics and predict stop distribution
        sent_topics = []
        for sent_i in range(config.MAX_SENT_NUM):
            sent_decoder_topic, sent_decoder_stop, sent_decoder_hidden = sent_decoder(sent_decoder_input,
                                                                                      sent_decoder_hidden)
            sent_topics.append(sent_decoder_topic)

            _stop_loss += criterion(sent_decoder_stop[0], stop_target_variable[sent_i])

        # generate sentences
        teacher_forcing = True if random.random() < 0.5 else False
        for sent_i in range(_sent_num):
            sent_target_variable = cap_target_variable[sent_i]

            _sent_len = (sent_target_variable == EOS_INDEX).nonzero().data[0][0] + 1
            _word_seen_num = 0

            word_decoder_hidden = sent_topics[sent_i] # init RNN using topic vector
            word_decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
            word_decoder_input = word_decoder_input.cuda() if torch.cuda.is_available() else word_decoder_input

            if teacher_forcing:
                for word_i in range(_sent_len):
                    word_decoder_output, word_decoder_hidden = word_decoder(word_decoder_input, word_decoder_hidden)
                    _cap_loss += criterion(word_decoder_output[0], sent_target_variable[word_i])
                    word_decoder_input = sent_target_variable[word_i]
                    _word_seen_num += 1
            else:
                for word_i in range(_sent_len):
                    word_decoder_output, word_decoder_hidden = word_decoder(word_decoder_input, word_decoder_hidden)
                    _cap_loss += criterion(word_decoder_output[0], sent_target_variable[word_i])
                    topv, topi = word_decoder_output[0].data.topk(1)
                    ni = topi[0][0]

                    word_decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
                    word_decoder_input = word_decoder_input.cuda() if torch.cuda.is_available() else word_decoder_input

                    _word_seen_num += 1
                    if ni == EOS_INDEX:
                        break
            _word_num += _word_seen_num

        return _seg_loss, _stop_loss, _cap_loss, _sent_num, _word_num
    
    batch_size = len(input_variables)
    for i in range(batch_size):
        cap_target_variable = cap_target_variables[i]
        seg_target_variable = seg_target_variables[i]
        stop_target_variable = stop_target_variables[i]
        input_variable = input_variables[i]

        cur_seg_loss, cur_stop_loss, cur_cap_loss, cur_sent_num, cur_word_num = _one_pass(input_variable, seg_target_variable, cap_target_variable,
                                                                            stop_target_variable, config.OnlySeg)

        total_sent_num += cur_sent_num
        total_word_num += cur_word_num

        stop_loss += cur_stop_loss
        cap_loss += cur_cap_loss
        seg_loss += cur_seg_loss

    if config.OnlySeg:
        loss = seg_loss
    else:
        loss = config.CapLoss_Weight * cap_loss + config.StopLoss_Weight * stop_loss + config.SegLoss_Weight * seg_loss
    loss.backward()

    encoder_optimizer.step()
    if not config.OnlySeg:
        sent_decoder_optimizer.step()
        word_decoder_optimizer.step()
        return stop_loss.data[0] / total_sent_num, cap_loss.data[0] / total_word_num, seg_loss.data[0] / batch_size
    else:
        return 0, 0, seg_loss.data[0] / batch_size


def as_minute(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since  # 已经经过的时间
    es = s / (percent)  # 估计的总时间
    rs = es - s  # 估计的剩余时间
    return '%s (- %s)' % (as_minute(s), as_minute(rs))


def show_plot(points, store_root, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig(os.path.join(store_root, name+'_tendency.png'))

def train_iters(encoder, sent_decoder, word_decoder, train_pairs, val_pairs, config, im_load_fn, start_iter):

    n_iters = config.NumIters
    batch_size = config.BatchSize
    print_every = config.PrintFrequency
    plot_every = config.PlotFrequency

    encoder_optimizer = torch.optim.SGD(params=encoder.parameters(), lr=config.LR, momentum=config.Momentum)
    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', patience=5, verbose=True)
    sent_decoder_optimizer, word_decoder_optimizer = None, None
    sent_decoder_scheduler, word_decoder_scheduler = None, None
    if not config.OnlySeg:
        sent_decoder_optimizer = torch.optim.SGD(params=sent_decoder.parameters(), lr=config.LR, momentum=config.Momentum)
        sent_decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sent_decoder_optimizer, mode='min', patience=5, verbose=True)
        word_decoder_optimizer = torch.optim.SGD(params=word_decoder.parameters(), lr=config.LR, momentum=config.Momentum)
        word_decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(word_decoder_optimizer, mode='min', patience=5, verbose=True)

    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()

    print_seg_loss_total = 0
    print_stop_loss_total = 0
    print_caption_loss_total = 0
    print_loss_total = 0

    plot_seg_loss_total = 0
    plot_stop_loss_total = 0
    plot_caption_loss_total = 0
    plot_loss_total = 0

    plot_losses = []
    plot_seg_losses = []
    plot_stop_losses = []
    plot_caption_losses = []

    for iter in range(start_iter, n_iters+start_iter):
        random.shuffle(train_pairs)
        dataset_index, batch_index, dataset_size = 0, 0, len(train_pairs)
        while dataset_index + batch_size < dataset_size:
            current_pairs = [variables_from_pair(lang, train_pairs[dataset_index+i], config.MAX_SENT_NUM, im_load_fn)
                              for i in range(batch_size)]

            dataset_index += batch_size
            batch_index += 1

            input_variables = [p[0] for p in current_pairs]
            seg_target_variables = [p[1] for p in current_pairs]
            cap_target_variables = [p[2] for p in current_pairs]
            stop_target_variables = [p[3] for p in current_pairs]

            stop_loss, caption_loss, seg_loss = train(input_variables, seg_target_variables, cap_target_variables, stop_target_variables, encoder,
                                            sent_decoder, word_decoder,encoder_optimizer, sent_decoder_optimizer,
                                            word_decoder_optimizer, criterion, config)
            # dynamic adjust the learning rate
            encoder_scheduler.step(seg_loss)

            loss = stop_loss + caption_loss + seg_loss
            if not config.OnlySeg:
                sent_decoder_scheduler.step(loss)
                word_decoder_scheduler.step(loss)

            print_loss_total += loss
            print_seg_loss_total += seg_loss
            print_stop_loss_total += stop_loss
            print_caption_loss_total += caption_loss

            plot_seg_loss_total += seg_loss
            plot_stop_loss_total += stop_loss
            plot_caption_loss_total += caption_loss
            plot_loss_total += loss

            if dataset_index % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_seg_loss_avg = print_seg_loss_total / print_every
                print_stop_loss_avg = print_stop_loss_total / print_every
                print_caption_loss_avg = print_caption_loss_total / print_every

                if config.OnlySeg:
                    seg_loss = evaluate_pairs(encoder, sent_decoder, word_decoder, val_pairs, config, n=10,
                                                 im_load_fn=im_load_fn)
                    print(
                        '[Iter: %d, Batch: %d]%s (%d %d%%) loss = %.3f, seg_loss = %.3f, (val) seg_loss = %.3f' %
                        (iter, batch_index, time_since(start, dataset_index / dataset_size), dataset_index,
                         dataset_index / dataset_size * 100,
                         print_loss_avg, print_seg_loss_avg, seg_loss))
                else:
                    bleu_scores = evaluate_pairs(encoder, sent_decoder, word_decoder, val_pairs, config, n=10, im_load_fn=im_load_fn)
                    print('[Iter: %d, Batch: %d]%s (%d %d%%) loss = %.3f, seg_loss = %.3f, stop_loss = %.3f, caption_loss = %.3f, bleu_score = [%.3f, %.3f, %.3f, %.3f]' %
                    (iter, batch_index, time_since(start, dataset_index / dataset_size), dataset_index, dataset_index / dataset_size * 100,
                                                print_loss_avg, print_seg_loss_avg, print_stop_loss_avg, print_caption_loss_avg, bleu_scores[0], bleu_scores[1], bleu_scores[2], bleu_scores[3]))

                print_loss_total, print_seg_loss_total, print_stop_loss_total, print_caption_loss_total = 0, 0, 0, 0

                if not config.OnlySeg:
                    display_randomly(encoder, sent_decoder, word_decoder, val_pairs, config, im_load_fn=im_load_fn)

            if batch_index % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_seg_loss_avg = plot_seg_loss_total / plot_every
                plot_stop_loss_avg = plot_stop_loss_total / plot_every
                plot_caption_loss_avg = plot_caption_loss_total /plot_every

                plot_losses.append(plot_loss_avg)
                plot_seg_losses.append(plot_seg_loss_avg)
                plot_stop_losses.append(plot_stop_loss_avg)
                plot_caption_losses.append(plot_caption_loss_avg)

                plot_loss_total = 0
                plot_seg_loss_total = 0
                plot_stop_loss_total = 0
                plot_caption_loss_total = 0

        if not config.OnlySeg:
            val_bleu_scores = evaluate_pairs(encoder, sent_decoder, word_decoder, val_pairs, config, im_load_fn=im_load_fn)
            print("[Iter {}] Validation BLEU: {:.3f} {:.3f} {:.3f} {:.3f}".format(iter, val_bleu_scores[0], val_bleu_scores[1], val_bleu_scores[2], val_bleu_scores[3]))

        save_model(encoder, sent_decoder, word_decoder, config, suffix='_'+str(iter))

    show_plot(plot_losses, config.store_root, name="loss")
    show_plot(plot_seg_losses, config.store_root, name='seg_loss')
    show_plot(plot_stop_losses, config.store_root, name="stop_loss")
    show_plot(plot_caption_losses, config.store_root, name="caption_loss")


# predict the caption for an image
def evaluate(encoder, sent_decoder, word_decoder, imagepath, config, im_load_fn):
    input_variable, seg_variable = variable_from_image_path(imagepath, load_fn=im_load_fn)

    # image representation
    im_embed, pred_seg = encoder(input_variable)
    np.save(os.path.join(config.StoreRoot, 'truth'), seg_variable.numpy())
    np.save(os.path.join(config.StoreRoot, 'pred'), pred_seg.data.numpy())

    if config.OnlySeg:
        criterion = torch.nn.CrossEntropyLoss()
        seg_loss = criterion(pred_seg, seg_variable)
        return seg_loss

    # generate sentence topics and stop distribution
    sent_decoder_hidden = sent_decoder.init_hidden()
    sent_decoder_input = im_embed
    topics = []
    for sent_i in range(config.MAX_SENT_NUM):
        sent_decoder_topic, sent_decoder_stop, sent_decoder_hidden = sent_decoder(sent_decoder_input, sent_decoder_hidden)
        # if it should stop in this step
        topv, topi = sent_decoder_stop.data.topk(1)
        ni = topi[0][0][0]
        if ni == 1:
            break

        topics.append(sent_decoder_topic)

    # generate each sentence
    decoded_caption = []
    for sent_i in range(len(topics)):
        topic_vec = topics[sent_i]

        # init word decoder by topic vector
        word_decoder_hidden = topic_vec
        word_decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
        word_decoder_input = word_decoder_input.cuda() if torch.cuda.is_available() else word_decoder_input

        decoded_sentence = []
        for word_i in range(config.MAX_WORD_NUM):
            word_decoder_output, word_decoder_hidden = word_decoder(word_decoder_input, word_decoder_hidden)
            topv, topi = word_decoder_output[0].data.topk(1)
            ni = topi[0][0]

            if ni == EOS_INDEX:
                break
            else:
                decoded_sentence.append(lang.idx2word[ni])

            word_decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
            word_decoder_input = word_decoder_input.cuda() if torch.cuda.is_available() else word_decoder_input
        decoded_caption.append(decoded_sentence)
    return decoded_caption


# randomly choose n images and predict their captions, store the resulted image
def evaluate_pairs(encoder, sent_decoder, word_decoder, pairs, config, im_load_fn, n=-1, verbose=False):
    store_path = os.path.join(config.StoreRoot, 'evaluation')
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    #from nltk.translate.bleu_score import sentence_bleu
    #from functools import reduce
    import random
    random.shuffle(val_pairs)
    if n > 0:
        pairs = pairs[:n]

    num = len(pairs)

    bleu_scores = []
    loss = 0
    for i in range(num):
        if verbose:
            print('{}/{}\r'.format(i, num), end='')
        pair = pairs[i]
        truth_cap = pair[1]
        if config.OnlySeg:
            seg_loss = evaluate(encoder, sent_decoder, word_decoder, pair[0], config, im_load_fn=im_load_fn)
            loss += seg_loss
            continue
        else:
            pred_cap = evaluate(encoder, sent_decoder, word_decoder, pair[0], config, im_load_fn=im_load_fn)

        truth = '。'.join(truth_cap)
        pred = '。'.join([''.join(sent) for sent in pred_cap])

        # segmentation
        import fool
        truth = fool.cut(truth)
        pred = fool.cut(pred)[0]
        # compute bleu
        import nltk
        cur_score_1 = nltk.translate.bleu(truth, pred, weights=[1,0,0,0])
        cur_score_2 = nltk.translate.bleu(truth, pred, weights=[0,1,0,0])
        cur_score_3 = nltk.translate.bleu(truth, pred, weights=[0,0,1,0])
        cur_score_4 = nltk.translate.bleu(truth, pred, weights=[0,0,0,1])
        bleu_scores.append([cur_score_1, cur_score_2, cur_score_3, cur_score_4])
        '''plt.figure()
        fig, ax = plt.subplots()
        im = np.array(Image.open(pair[0]))
        plt.imshow(im)
        plt.title('%s\nGT:%s' % (truth_cap, pred_cap))
        plt.axis('off')
        plt.savefig(os.path.join(store_path, str(i)+'.png'))'''
    if not config.OnlySeg:
        return np.mean(np.array(bleu_scores), axis=0)
    else:
        return loss / num

def display_randomly(encoder, sent_decoder, word_decoder, val_pairs, config, im_load_fn):
    pair = random.choice(val_pairs)
    truth_cap = pair[1]
    print("Truth: ", '。'.join(truth_cap))
    pred_cap = evaluate(encoder, sent_decoder, word_decoder, pair[0], config, im_load_fn=im_load_fn)
    print("Prediction:", '。'.join([''.join(sent) for sent in pred_cap]))


########################
# Metrics:
# Use pycocoevalcap ( https://github.com/kelvinxu/arctic-captions/blob/master/metrics.py) -> python 3
########################

class Metrics:

    def __init__(self):
        from coco_caption.pycocoevalcap.bleu.bleu import Bleu
        from coco_caption.pycocoevalcap.cider.cider import Cider
        from coco_caption.pycocoevalcap.rouge.rouge import Rouge
        from coco_caption.pycocoevalcap.meteor.meteor import Meteor

        self.bleu = Bleu()
        self.cider = Cider()
        self.rouge = Rouge()
        self.meteor = Meteor()


    def compute_single_score(self, truth, pred):
        '''
        Computer several metrics
        :param truth: <String> the ground truth sentence
        :param pred:  <String> predicted sentence
        :return: score list
        '''
        bleu_gts = {'1': [truth]}
        bleu_res = {'1': [pred]}
        bleu_score = self.bleu.compute_score(bleu_gts, bleu_res)

        rouge_gts = bleu_gts
        rouge_res = bleu_res
        rouge_score = self.rouge.compute_score(rouge_gts, rouge_res)

        return {'BLEU': bleu_score[0], 'ROUGE': rouge_score[0]}

    def compute_set_score(self, truths, preds):
        gts = {k: [v] for k, v in truths.items()}
        res = {k: [v] for k, v in preds.items()}

        bleu_score = self.bleu.compute_score(gts, res)
        rouge_score = self.rouge.compute_score(gts, res)
        cider_score = self.cider.compute_score(gts, res)

        return {'BLEU': bleu_score[0], 'ROUGE': rouge_score[0], 'CIDEr': cider_score[0]}


#########################
# Configuration
########################

torch.manual_seed(1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Medical Captioning")
    parser.add_argument('--im', required=False,
                        default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/dataset/BRATS/images',
                        metavar="path/to/image/dataset",
                        help="The image dataset")
    parser.add_argument('--trainval-cap', required=False,
                        default="/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/dataset/BRATS/train_captions.txt",
                        metavar='path/to/trainval/findings',
                        help="The medical image captions for training and validation")
    parser.add_argument('--test-cap', required=False,
                        default="/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/dataset/BRATS/test_captions.txt",
                        metavar="path/to/test/findings",
                        help='The medical image captions for testing')
    parser.add_argument('--store-root', required=False,
                        default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/MedCap/checkpoints',
                        # "/mnt/md1/lztao/models/med_cap",
                        metavar='path/to/store/models',
                        help="Store model")
    parser.add_argument('--load-root', required=False, default=None,
                        metavar='path/to/saved/models',
                        help="the path to models for restore.")
    parser.add_argument('--val-prop', required=False, default=0.1,
                        metavar='proportionate of validation dataset')
    parser.add_argument('--seg-mode', required=False, default='word',
                        metavar='how to conduct word segmentation')
    parser.add_argument('--start-iter', required=False, default=0,
                        metavar='the start iteration number')
    args = parser.parse_args()
    print("Arguments: ")
    print("Image Dataset: ", args.im)
    print("Caption Dataset (trainval): ", args.trainval_cap)
    print("Caption Dataset (test): ", args.test_cap)
    print("Validation Proportionate: ", args.val_prop)
    print("Store root: ", args.store_root)

    if args.load_root is not None:
        print("Loading from last experiment settings...")
        [train_pairs, val_pairs, lang] = pickle.load(open(os.path.join(args.store_root, 'exp_config.pkl'), 'rb'))
    else:
        print("\nRead Captions....")
        trainval_pairs = read_captions(args.trainval_cap, args.im)

        train_pairs, val_pairs = split_train_val(trainval_pairs, val_prop=args.val_prop)

        lang = prepare_dict(train_pairs, mode=args.seg_mode)

    test_pairs = read_captions(args.test_cap, args.im)

    stat_lang(lang)
    print("Train samples: ", len(train_pairs))
    print("Validation samples: ", len(val_pairs))
    print("Test samples: ", len(test_pairs))

    random.shuffle(train_pairs)
    stat_captions(train_pairs)

    # store the configuration
    pickle.dump([train_pairs, val_pairs, lang], open(os.path.join(args.store_root, 'exp_config.pkl'), 'wb'))


    class IUChest_Config(Config):
        StoreRoot = args.store_root

        # the maximum number of sentences and maximum number of words per sentence
        MAX_SENT_NUM = 15
        MAX_WORD_NUM = 50

        # dictionary size
        DICT_SIZE = len(lang)

        # Shape of feature map extracted from CNN
        FeatureShape = (1024, 15, 15)

        # Train Configuration
        OnlySeg = True

        def __int__(self):
            super(IUChest_Config, self).__init__()

    config = IUChest_Config()
    config.display()

    print("Create model...")
    encoder = Encoder(config)
    sent_decoder, word_decoder = None, None
    if not config.OnlySeg:
        sent_decoder = SentDecoder(config)
        word_decoder = WordDecoder(config)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        sent_decoder = sent_decoder.cuda() if sent_decoder is not None else None
        word_decoder = word_decoder.cuda() if word_decoder is not None else None

    if args.load_root:
        try:
            load_model(encoder, sent_decoder, word_decoder, args.load_root)
        except KeyError:
            print("The model file is invalid. \nExit!")
            import sys
            sys.exit(0)


    print("--------Train--------")
    train_iters(encoder, sent_decoder, word_decoder, train_pairs, val_pairs, config, im_load_fn=np.load, start_iter=args.start_iter)

    print("--------Evaluate--------")
    sentences = evaluate(encoder, sent_decoder, word_decoder, config.TestImagePath, config, np.load)
    print('. '.join([' '.join(sent) for sent in sentences]))

    print("--------Test--------")
    evaluate_pairs(encoder, sent_decoder, word_decoder, test_pairs, config, verbose=True, im_load_fn=np.load)
