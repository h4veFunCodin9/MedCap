import math
import time, os

import matplotlib
# https://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab/4706614#4706614
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
import numpy as np

##########################
# Time log
#########################
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

###########################
# Visualize
###########################
def show_plot(points, store_root, name):
    plot_root = os.path.join(store_root, 'loss')
    if not os.path.exists(plot_root):
        os.makedirs(plot_root)

    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig(os.path.join(plot_root, name+'_tendency.png'))
    return

def display_caption(image_path, truth_cap, pred_cap, store_path):
    plt.figure()
    fig, ax = plt.subplots()
    im = np.array(Image.open(image_path))
    plt.imshow(im)
    plt.title('%s\nGT:%s' % (truth_cap, pred_cap))
    plt.axis('off')
    plt.savefig(store_path)

###########################
# Metrics
###########################
class BLEUCalculate():
    def __init__(self):
        self.score1 = 0.0
        self.score2 = 0.0
        self.score3 = 0.0
        self.score4 = 0.0
        self.n = 0

    def add(self, truth, pred):
        # segmentation
        truth_words = []
        for sent in truth:
            truth_words.extend(sent)
            truth_words.append('。')
        pred_words = []
        for sent in pred:
            pred_words.extend(sent)
            pred_words.append('。')

        # compute bleu
        import nltk
        self.score1 += nltk.translate.bleu(truth_words, pred_words, weights=[1, 0, 0, 0])
        self.score2 += nltk.translate.bleu(truth_words, pred_words, weights=[0, 1, 0, 0])
        self.score3 += nltk.translate.bleu(truth_words, pred_words, weights=[0, 0, 1, 0])
        self.score4 += nltk.translate.bleu(truth_words, pred_words, weights=[0, 0, 0, 1])
        self.n += 1

    def get_scores(self):
        if self.n == 0:
            return [-1, -1, -1, -1]
        return [self.score1 / self.n, self.score2 / self.n, self.score3 / self.n, self.score3 / self.n, self.score4 / self.n]


class IOUCalculate():
    def __init__(self):
        self.inter = [0.0] * 4
        self.union = [0.0] * 4
        self.epsilon = 0.0001

    def add(self, truth, pred):
        pred = np.argmax(pred, axis=1).astype(np.int16)

        truth = [truth == 1, truth == 2, truth == 3, truth != 0]
        pred = [pred == 1, pred == 2, pred == 3, pred != 0]

        for i in range(4):
            self.inter[i] += np.sum(np.logical_and(truth[i], pred[i]))
            self.union[i] += np.sum(np.logical_or(truth[i], pred[i]))

    def get_iou(self):
        return [(self.inter[i] + self.epsilon) / (self.union[i] + self.epsilon) for i in range(4)]
