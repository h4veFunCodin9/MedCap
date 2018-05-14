import numpy as np
import fool
fool.load_userdict('data/foolnltk_userdict')
from functools import reduce

SOS_INDEX = 0
EOS_INDEX = 1


class Lang:
    '''
    The dictionary for a language
    '''
    def __init__(self, name, mode='word'):
        self.name = name
        self.mode = mode
        self.word2idx = {}
        self.idx2word = {0:'SOS', 1:'EOS'}
        self.word2count = {}
        self.n_words = 2
        self.word2weight = {}

    def addSentence(self, s):
        if self.mode == 'char':
            terms = list(s)
        elif self.mode == 'word':
            terms = fool.cut(s)
        else:
            print("Unknow mode {}.".format(self.mode))
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

    def assign_weight(self):
        total_occur = 0
        for word, count in self.word2count.items():
            total_occur += count
        mean_occur = total_occur / len(self.word2count)

        for word, count in self.word2count.items():
            self.word2weight[word] = mean_occur / count

    def __len__(self):
        return self.n_words

    def stat(self):
        print("Language Dictionary: {} words.".format(str(self.n_words)))

        stat = self.word2count.copy()
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