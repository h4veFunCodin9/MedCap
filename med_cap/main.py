import torch
import torchvision.models as M
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg') # https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab/4706614#4706614
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PIL import Image
import numpy as np
import unicodedata
import re, random
import time, math
import os

import skimage.transform as T

from config import Config

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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1 ", s)   #将标点符号用空格分开
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  #除字母标点符号的其他连续字符替换成一个空格
    return s

def readCaptions(annFile, image_root):
    dataset = open(annFile, 'r').read()
    dataset = dataset.split('\n')
    pairs = []
    for p_str in dataset:
        if len(p_str) <= 1:
            continue
        image_id, caption = p_str.split('\t')
        caption = normalizeString(caption)
        caption = [sent.strip() for sent in caption.split(' .') if len(sent.strip()) > 0]
        pairs.append((os.path.join(image_root, image_id+'.png'), caption))
    return pairs

def prepareDict(pairs):
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


'''
Model: Encoder and Decoder
'''
def variableFromImagePath(image_path):
    im = np.array(Image.open(image_path))
    im = T.resize(im, (512, 512, 3), mode='reflect')
        
    data = np.zeros([1, 3, 512, 512]) # IU chest X-Ray (COCO 640x480)
    data[0, 0, ...], data[0, 1, ...], data[0, 2, ...] = im[:,:,0], im[:,:,1], im[:,:,2]
    
    data = torch.autograd.Variable(torch.FloatTensor(data))
    data = data.cuda() if torch.cuda.is_available() else data
    return data

def variableFromCaption(lang, cap):
    indices = [[lang.word2idx[w] for w in sent.split(' ')]for sent in cap]
    max_len = max([len(sent) for sent in indices])
    # append End_Of_Sequence token; increase to the same size
    for sent in indices:
        sent.extend([EOS_INDEX]*(max_len-len(sent)+1))
    indices = torch.autograd.Variable(torch.LongTensor(indices)).view(-1, max_len+1)
    return indices.cuda() if torch.cuda.is_available() else indices

def variablesFromPair(lang, pair):
    image_var = variableFromImagePath(pair[0])
    cap_var = variableFromCaption(lang, pair[1])
    return image_var, cap_var

# hidden_size = embedding size

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

# TODO
class SentDecoder(torch.nn.Module):
    def __init__(self, input_size, config):
        super(SentDecoder, self).__init__()
        self.hidden_size = config.SentLSTM_HiddenSize
        self.topic_size = config.TopicSize
        # context vector for each time step
        self.ctx_im_W = torch.nn.Linear(input_size, self.hidden_size)
        self.ctx_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # RNN unit
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        # topic output
        self.topic_h_W = torch.nn.Linear(self.hidden_size, self.topic_size)
        self.topic_ctx_W = torch.nn.Linear(self.hidden_size, self.topic_size)
        # stop distribution output
        self.stop_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_prev_h_W = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.stop_W = torch.nn.Linear(self.hidden_size, 1)

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

        return topic, stop

class WordDecoder(torch.nn.Module):
    def __init__(self, config):
        super(WordDecoder, self).__init__()

class Decoder(torch.nn.Module):
    def __init__(self, input_size, config):
        super(Decoder, self).__init__()
        self.hidden_size = config.HiddenSize

        self.embedding = torch.nn.Embedding(input_size, config.HiddenSize)
        self.gru = torch.nn.GRU(config.HiddenSize, config.HiddenSize)
        self.out = torch.nn.Linear(config.HiddenSize, input_size)
        #self.sfm = torch.nn.Softmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        #output = self.sfm(out)
        return output, hidden


'''
Train and evaluate functionality
'''

def train(input_variables, target_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=50):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    total_len = 0

    def one_pass(input_variable, target_variable):
        im_embedding = encoder(input_variable) # [1, HiddenSize]

        target_len = target_variable.size()[0]

        sent_loss = 0
        stop_loss = 0

        decoder_hidden = im_embedding.view(1, 1, -1)
        decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
        decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

        teacher_forcing = True if random.random() < 0.5 else False
        if teacher_forcing:
            for di in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                sent_loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
        else:
            for di in range(target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                sent_loss += criterion(decoder_output, target_variable[di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
                decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
                if ni == EOS_INDEX:
                    break
        return sent_loss, target_len

    batch_size = len(target_variables)
    for i in range(batch_size):
        target_variable = target_variables[i]
        input_variable = input_variables[i]
        cur_loss, cur_len = one_pass(input_variable, target_variable)
        total_len += cur_len
        loss += cur_loss

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / total_len

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since  # 已经经过的时间
    es = s / (percent)  # 估计的总时间
    rs = es - s  # 估计的剩余时间
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig('loss_tendency.png')

def save_model(encoder, decoder, store_root, info):
    print('saving model...')
    iter, bleu = info
    # save the model
    torch.save(encoder.state_dict(), os.path.join(store_root, 'encoder'))#'_'.join(["encoder", str(iter), '{:.3f}'.format(bleu)])))
    torch.save(decoder.state_dict(), os.path.join(store_root, 'decoder'))#'_'.join(["decoder", str(iter), '{:.3f}'.format(bleu)])))
    print('done')

def trainIters(encoder, decoder, config, epoch, batch_size=4, print_every=1, plot_every=100):

    encoder_optimizer = torch.optim.SGD(params=encoder.parameters(), lr=config.LR, momentum=config.Momentum)
    decoder_optimizer = torch.optim.SGD(params=decoder.parameters(), lr=config.LR, momentum=config.Momentum)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()

    print_loss_total = 0
    plot_loss_total = 0

    plot_losses = []

    for iter in range(1, epoch+1):
        random.shuffle(pairs)
        dataset_index, batch_index, dataset_size = 0, 0, len(pairs)
        while dataset_index + batch_size < dataset_size:
            training_pairs = [variablesFromPair(lang, pairs[dataset_index+i]) for i in range(batch_size)]

            dataset_index += batch_size
            batch_index += 1

            input_variables = [p[0] for p in training_pairs]
            target_variables = [p[1] for p in training_pairs]

            loss = train(input_variables, target_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if batch_index % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                bleu = evaluateRandomly(encoder, decoder, config.StoreRoot)
                print('%s (%d %d%%) %.4f; bleu = %.4f' % (timeSince(start, dataset_index / dataset_size), dataset_index, dataset_index / dataset_size * 100, print_loss_avg, bleu))
                print_loss_total = 0
                displayRandomly(encoder, decoder)
<<<<<<< HEAD
                save_model(encoder, decoder, config.StoreRoot, (iter, bleu))

=======


>>>>>>> b127597b519a7abb03239b3b46a7008983f870fb
            if batch_index % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)

# predict the caption for an image
def evaluate(encoder, decoder, imagepath, max_length=50):
    input_variable = variableFromImagePath(imagepath)

    im_embed = encoder(input_variable)

    decoder_hidden = im_embed.view(1, 1, -1)
    decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
    decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if ni == EOS_INDEX:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.idx2word[ni])

        decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
        decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

    return decoded_words

# randomly choose n images and predict their captions, store the resulted image
def evaluateRandomly(encoder, decoder, store_dir, n=50, n_save_image=10):
    store_path = os.path.join(store_dir, 'evaluation')
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    sampled_pairs = [random.choice(pairs) for _ in range(n)]
    random.shuffle(sampled_pairs)

    from nltk.translate.bleu_score import sentence_bleu
    bleu = 0
    for i in range(n):
        pair = sampled_pairs[i]
        im = np.array(Image.open(pair[0]))
        truth_cap = pair[1]
        pred_cap = evaluate(encoder, decoder, pair[0])
        bleu += sentence_bleu([truth_cap.split(' ')], pred_cap)

        if i < n_save_image:
            plt.figure()
            fig, ax = plt.subplots()

            plt.imshow(im)
            plt.title('%s\nGT:%s' % (truth_cap, pred_cap))
            plt.axis('off')
            plt.savefig(os.path.join(store_path, str(i) + '.png'))

            plt.close(fig)

    return bleu / n

def displayRandomly(encoder, decoder):
    pair = random.choice(pairs)
    truth_cap = pair[1]
    print("Truth: ", truth_cap)
    pred_cap = evaluate(encoder, decoder, pair[0])
    print("Prediction:", end=' ')
    for w in pred_cap:
        print(w, end=' ')
    print('')
'''
Train and evaluate
'''
'''import argparse
parser = argparse.ArgumentParser(description="parse command line arguments")
parser.add_argument('--lr', type=float, default=1.0e-3)

args = parser.parse_args()
'''
#########################
# Configuration
########################

torch.manual_seed(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Medical Captioning")
<<<<<<< HEAD
    parser.add_argument('--im', required=False, default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/NLMCXR_png', #'/mnt/md1/lztao/dataset/IU_Chest_XRay/NLMCXR_png',
                        metavar="path/to/image/dataset",
                        help="The image dataset")
    parser.add_argument('--cap', required=False, default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/findings.txt', #"mnt/md1/lztao/dataset/IU_Chest_XRay/findings.txt",
=======
    parser.add_argument('--im', required=False, default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/NLMCXR_png',
                        metavar="path/to/image/dataset",
                        help="The image dataset")
    parser.add_argument('--cap', required=False, default="/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/findings.txt",
>>>>>>> b127597b519a7abb03239b3b46a7008983f870fb
                        metavar='path/to/findings',
                        help="The medical image captions")
    parser.add_argument('--store-root', required=False, default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/MedCap/checkpoints', #"/mnt/md1/lztao/models/med_cap",
                        metavar='path/to/store/models',
                        help="Store model")
    args = parser.parse_args()


    class IUChest_Config(Config):
        StoreRoot = args.store_root

        # the maximum number of sentences and maximum number of words per sentence
        MAX_SENT_NUM = 20
        MAX_WORD_NUM = 45


    config = IUChest_Config()

    print("Loading dataset...")
    pairs = readCaptions(args.cap, args.im)
    lang = prepareDict(pairs)
    stat_lang(lang)
    print("Training samples: ", len(pairs))
    random.shuffle(pairs)

<<<<<<< HEAD
    print("Create model...")
=======
    stat_captions(pairs)

    print(variablesFromPair(lang, pairs[0]))
    input()

>>>>>>> b127597b519a7abb03239b3b46a7008983f870fb
    encoder = Encoder(config)
    decoder = Decoder(lang.n_words, config)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("------Train-----")
    trainIters(encoder, decoder, config, 100, batch_size=4, print_every=1, plot_every=10)
    print("------Evaluate----")
    words = evaluate(encoder, decoder, config.TestImagePath)
    for w in words:
        print(w, end=' ')
