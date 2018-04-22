import torch
import torchvision.models as M
import torch.nn.functional as F

from PIL import Image
import numpy as np
import unicodedata
import re, random
import time, math
import os

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
        self.n_words = 3

    def addSentence(self, s):
        for w in s.split(' '):
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
    s = re.sub(r"([.!?])", r" \1 ", s)   #将标点符号用空格分开
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  #除字母标点符号的其他连续字符替换成一个空格
    return s


def read_captions(annFile, image_root):
    dataset = open(annFile, 'r').read()
    dataset = dataset.split('\n')
    pairs = []
    for p_str in dataset:
        if len(p_str) <= 1:
            continue
        image_id, caption = p_str.split('\t')
        caption = normalize_string(caption)
        caption = [sent.strip() for sent in caption.split(' .') if len(sent.strip()) > 0]
        pairs.append((os.path.join(image_root, image_id+'.png'), caption))
    return pairs


def prepare_dict(pairs):
    lang = Lang("eng")
    for pair in pairs:
        for sent in pair[1]:
            lang.addSentence(sent)
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
def stat_captions(pairs):
    w_max, s_max = 0, 0
    for path, caption in pairs:
        s_max = s_max if len(caption) < s_max else len(caption)
        for sent in caption:
            w_max = w_max if len(sent.split(' ')) < w_max else len(sent.split(' '))
    print("Max number of sentences: ", s_max)
    print("Max length of sentences: ", w_max)


def variable_from_image_path(image_path):
    im = np.array(Image.open(image_path))
    im = T.resize(im, (512, 512, 3), mode='reflect')

    data = np.zeros([1, 3, 512, 512])  # IU chest X-Ray (COCO 640x480)
    data[0, 0, ...], data[0, 1, ...], data[0, 2, ...] = im[:, :, 0], im[:, :, 1], im[:, :, 2]

    data = torch.autograd.Variable(torch.FloatTensor(data))
    data = data.cuda() if torch.cuda.is_available() else data
    return data


def variable_from_caption(lang, cap, max_sent_num):
    indices = [[lang.word2idx[w] for w in sent.split(' ')] for sent in cap]
    stop = [0 if i < len(indices) else 1 for i in range(max_sent_num)]

    max_len = max([len(sent) for sent in indices])
    # append End_Of_Sequence token; increase to the same size
    for sent in indices:
        sent.extend([EOS_INDEX] * (max_len - len(sent) + 1))

    indices = torch.autograd.Variable(torch.LongTensor(indices)).view(-1, max_len + 1)
    stop = torch.autograd.Variable(torch.LongTensor(stop)).view(-1, 1)
    return indices.cuda() if torch.cuda.is_available() else indices, stop.cuda() if torch.cuda.is_available() else stop


def variables_from_pair(lang, pair, max_sent_num):
    image_var = variable_from_image_path(pair[0])
    cap_var, stop_var = variable_from_caption(lang, pair[1], max_sent_num)
    return image_var, cap_var, stop_var


#############################
# Model: Encoder and Decoder
##############################
class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding_size = config.IM_EmbeddingSize

        self.vgg = M.vgg11(pretrained=False)
        shape = config.FeatureShape
        self.linear = torch.nn.Linear(in_features=(shape[0] * shape[1] * shape[2]), out_features=self.embedding_size)

    def forward(self, x):   # IU Chest X-ray image: [1, 3, 512, 512]
        feature = self.vgg.features(x) # IU Chest X-ray image: [1, 512, 16, 16]
        embedding = self.linear(feature.view(-1))
        return embedding.view(1, -1)



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

    def forward(self, input, hidden):
        x = input
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

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


def save_model(encoder, sent_decoder, word_decoder, store_root):
    # save the model
    print("Saving models...")
    torch.save(encoder.state_dict(), os.path.join(store_root, "encoder"))
    torch.save(sent_decoder.state_dict(), os.path.join(store_root, "sent_decoder"))
    torch.save(word_decoder.state_dict(), os.path.join(store_root, "word_decoder"))
    print("Done!")


def load_model(encoder, sent_decoder, word_decoder, load_root):
    print("Loading models from '{}' ...".format(load_root))
    encoder_path = os.path.join(load_root, 'encoder')
    sent_decoder_path = os.path.join(load_root, 'sent_decoder')
    word_decoder_path = os.path.join(load_root, 'word_decoder')

    encoder.load_state_dict(torch.load(encoder_path))
    sent_decoder.load_state_dict(torch.load(sent_decoder_path))
    word_decoder.load_state_dict(torch.load(word_decoder_path))
    print("Loaded.")


'''
Train and evaluate functionality
'''


def train(input_variables, cap_target_variables, stop_target_variables, encoder, sent_decoder, word_decoder,
          encoder_optimizer, sent_decoder_optimizer, word_decoder_optimizer, criterion, config):

    encoder_optimizer.zero_grad()
    sent_decoder_optimizer.zero_grad()
    word_decoder_optimizer.zero_grad()

    stop_loss, cap_loss = 0, 0

    total_sent_num, total_word_num = 0, 0

    # input_variable: 1 x 3 x 512 x 512, cap_target_variable: num_sent x len_sent, stop_target_variable: max_num_sent x 1
    def _one_pass(input_variable, cap_target_variable, stop_target_variable):
        _im_embedding = encoder(input_variable) # [1, HiddenSize]

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
                    word_decoder_input = word_decoder_input.cuda if torch.cuda.is_available() else word_decoder_input

                    _word_seen_num += 1
                    if ni == EOS_INDEX:
                        break
            _word_num += _word_seen_num

        return _stop_loss, _cap_loss, _sent_num, _word_num
    
    batch_size = len(input_variables)
    for i in range(batch_size):
        cap_target_variable = cap_target_variables[i]
        stop_target_variable = stop_target_variables[i]
        input_variable = input_variables[i]

        cur_stop_loss, cur_cap_loss, cur_sent_num, cur_word_num = _one_pass(input_variable, cap_target_variable,
                                                                            stop_target_variable)

        total_sent_num += cur_sent_num
        total_word_num += cur_word_num

        stop_loss += cur_stop_loss
        cap_loss += cur_cap_loss

    loss = config.CapLoss_Weight * cap_loss + config.StopLoss_Weight * stop_loss
    loss.backward()

    encoder_optimizer.step()
    sent_decoder_optimizer.step()
    word_decoder_optimizer.step()

    return stop_loss.data[0] / total_sent_num, cap_loss.data[0] / total_word_num,


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


def show_plot(points, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig(name+'_tendency.png')


def train_iters(encoder, sent_decoder, word_decoder, config):

    n_iters = config.NumIters
    batch_size = config.BatchSize
    print_every = config.PrintFrequency
    plot_every = config.PlotFrequency

    encoder_optimizer = torch.optim.SGD(params=encoder.parameters(), lr=config.LR, momentum=config.Momentum)
    sent_decoder_optimizer = torch.optim.SGD(params=sent_decoder.parameters(), lr=config.LR, momentum=config.Momentum)
    word_decoder_optimizer = torch.optim.SGD(params=word_decoder.parameters(), lr=config.LR, momentum=config.Momentum)

    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()

    print_stop_loss_total = 0
    print_caption_loss_total = 0
    print_loss_total = 0

    plot_stop_loss_total = 0
    plot_caption_loss_total = 0
    plot_loss_total = 0

    plot_losses = []
    plot_stop_losses = []
    plot_caption_losses = []

    for iter in range(1, n_iters+1):
        random.shuffle(pairs)
        dataset_index, batch_index, dataset_size = 0, 0, len(pairs)
        while dataset_index + batch_size < dataset_size:
            training_pairs = [variables_from_pair(lang, pairs[dataset_index+i], config.MAX_SENT_NUM)
                              for i in range(batch_size)]

            dataset_index += batch_size
            batch_index += 1

            input_variables = [p[0] for p in training_pairs]
            cap_target_variables = [p[1] for p in training_pairs]
            stop_target_variables = [p[2] for p in training_pairs]

            stop_loss, caption_loss = train(input_variables, cap_target_variables, stop_target_variables, encoder,
                                            sent_decoder, word_decoder,encoder_optimizer, sent_decoder_optimizer,
                                            word_decoder_optimizer, criterion, config)
            loss = stop_loss + caption_loss
            print_loss_total += loss
            print_stop_loss_total += stop_loss
            print_caption_loss_total += caption_loss

            plot_stop_loss_total += stop_loss
            plot_caption_loss_total += caption_loss
            plot_loss_total += loss

            if batch_index % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_stop_loss_avg = print_stop_loss_total / print_every
                print_caption_loss_avg = print_caption_loss_total / print_every

                # TODO evaluate validation dataset
                bleu = evaluate_randomly(encoder, sent_decoder, word_decoder, config)

                print('%s (%d %d%%) loss = %.3f, stop_loss = %.3f, caption_loss = %.3f; bleu = %.4f' %
                    (time_since(start, dataset_index / dataset_size), dataset_index, dataset_index / dataset_size * 100,
                                                print_loss_avg, print_stop_loss_avg, print_caption_loss_avg, bleu))

                print_loss_total, print_stop_loss_total, print_caption_loss_total = 0, 0, 0

                display_randomly(encoder, sent_decoder, word_decoder, config)
                # TODO save the best model on validate dataset
                save_model(encoder, sent_decoder, word_decoder, config.StoreRoot)

            if batch_index % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_stop_loss_avg = plot_stop_loss_total / plot_every
                plot_caption_loss_avg = plot_caption_loss_total /plot_every

                plot_losses.append(plot_loss_avg)
                plot_stop_losses.append(plot_stop_loss_avg)
                plot_caption_losses.append(plot_caption_loss_avg)

                plot_loss_total = 0
                plot_stop_loss_total = 0
                plot_caption_loss_total = 0

    show_plot(plot_losses, name="loss")
    show_plot(plot_stop_losses, name="stop_loss")
    show_plot(plot_caption_losses, name="caption_loss")


# predict the caption for an image
def evaluate(encoder, sent_decoder, word_decoder, imagepath, config):
    input_variable = variable_from_image_path(imagepath)

    # image representation
    im_embed = encoder(input_variable)

    # generate sentence topics and stop distribution
    sent_decoder_hidden = sent_decoder.init_hidden()
    sent_decoder_input = im_embed
    topics = []
    for sent_i in range(config.MAX_SENT_NUM):
        sent_decoder_topic, sent_decoder_stop, sent_decoder_hidden = sent_decoder(sent_decoder_input, sent_decoder_hidden)

        # if it should stop in this step
        topv, topi = sent_decoder_stop.data.topk(1)
        ni = topi[0][0]
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
                decoded_sentence.append('<EOS>')
                break
            else:
                decoded_sentence.append(lang.idx2word[ni])

            word_decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
            word_decoder_input = word_decoder_input.cuda() if torch.cuda.is_available() else word_decoder_input
        decoded_caption.append(decoded_sentence)
    return decoded_caption


# randomly choose n images and predict their captions, store the resulted image
def evaluate_randomly(encoder, sent_decoder, word_decoder, config, n=1):
    store_path = os.path.join(config.StoreRoot, 'evaluation')
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    from nltk.translate.bleu_score import sentence_bleu
    from functools import reduce
    bleu = 0
    for i in range(n):
        pair = random.choice(pairs)
        im = np.array(Image.open(pair[0]))
        truth_cap = pair[1]
        pred_cap = evaluate(encoder, sent_decoder, word_decoder, pair[0], config)
        #print(' '.join(truth_cap).split(' '))
        #print(reduce(lambda x, y: x + y, pred_cap))
        bleu += sentence_bleu(' '.join(truth_cap).split(' '), reduce(lambda x, y: x + y, pred_cap))
        # TODO: Using pycocoevalcap ( https://github.com/kelvinxu/arctic-captions/blob/master/metrics.py) -> python 3


        plt.figure()
        fig, ax = plt.subplots()

        plt.imshow(im)
        plt.title('%s\nGT:%s' % (truth_cap, pred_cap))
        plt.axis('off')
        plt.savefig(os.path.join(store_path, str(i)+'.png'))
    return bleu / n


def display_randomly(encoder, sent_decoder, word_decoder, config):
    pair = random.choice(pairs)
    truth_cap = pair[1]
    print("Truth: ", '. '.join(truth_cap))
    pred_cap = evaluate(encoder, sent_decoder, word_decoder, pair[0], config)
    print("Prediction:", '. '.join([' '.join(sent) for sent in pred_cap]))

#########################
# Configuration
########################

torch.manual_seed(1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Medical Captioning")
    parser.add_argument('--im', required=False, default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/NLMCXR_png',
                        metavar="path/to/image/dataset",
                        help="The image dataset")
    # TODO add trainval and test dataset
    parser.add_argument('--cap', required=False, default="/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/findings.txt",
                        metavar='path/to/findings',
                        help="The medical image captions")
    parser.add_argument('--store-root', required=False, default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/MedCap/checkpoints', #"/mnt/md1/lztao/models/med_cap",
                        metavar='path/to/store/models',
                        help="Store model")
    parser.add_argument('--load-root', required=False, default='.',
                        metavar='path/to/saved/models',
                        help="the path to models for restore.")
    args = parser.parse_args()
    print("Image Dataset: ", args.im)
    print("Caption Dataset: ", args.cap)
    print("Store root: ", args.store_root)

    print("\nRead Captions....")
    pairs = read_captions(args.cap, args.im)
    lang = prepare_dict(pairs)
    stat_lang(lang)
    print("Training samples: ", len(pairs))
    random.shuffle(pairs)
    stat_captions(pairs)

    class IUChest_Config(Config):
        StoreRoot = args.store_root

        # the maximum number of sentences and maximum number of words per sentence
        MAX_SENT_NUM = 20
        MAX_WORD_NUM = 45

        # dictionary size
        DICT_SIZE = len(lang)

        def __int__(self):
            super(IUChest_Config, self).__init__()

    config = IUChest_Config()
    config.display()

    print("Create model...")
    encoder = Encoder(config)
    sent_decoder = SentDecoder(config)
    word_decoder = WordDecoder(config)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        sent_decoder = sent_decoder.cuda()
        word_decoder = word_decoder.cuda()

    if os.path.isfile(args.load_root):
        print("Loading model from {}.".format(args.load_root))
        try:
            load_model(encoder, sent_decoder, word_decoder, args.load_root)
            print("Loaded!")
        except KeyError:
            print("The model file is invalid. \nExit!")
            import sys
            sys.exit(0)


    print("--------Train--------")
    train_iters(encoder, sent_decoder, word_decoder, config)

    # TODO test on test dataset when training is finished.

    print("--------Evaluate--------")
    sentences = evaluate(encoder, sent_decoder, word_decoder, config.TestImagePath)
    print('. '.join([' '.join(sent) for sent in sentences]))
