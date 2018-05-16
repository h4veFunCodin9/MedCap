import re
import unicodedata
import os
import copy
from .language import Lang

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.strip())
    #s = re.sub(r"([。*])", r" \1 ", s)   #将标点符号用空格分开
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  #除字母标点符号的其他连续字符替换成一个空格
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
        caption = [[word for word in sent.split() if len(word) > 0] for sent in caption.split(' 。 ') if len(sent.strip()) > 0]
        pairs.append((os.path.join(image_root, image_id+'.npy'), caption))
    return pairs

def generate_lang(ds):
    lang = Lang("ch", mode=ds.lang_mode)
    for pair in ds.pairs:
        for sent in pair[1]:
            lang.addSentence(sent)
    #print("#{}# Language Dictionary: {} words.".format(self.name, str(lang.n_words)))
    return lang