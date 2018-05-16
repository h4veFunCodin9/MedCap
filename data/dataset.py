from . import utils
from .language import Lang, SOS_INDEX, EOS_INDEX
import os
import torch
import numpy as np

class Dataset():
    def __init__(self, name, annFile=None, image_root=None, pairs=None, mode='word',
                 load_fn=None):
        if mode not in ['word', 'char']:
            print('Dataset: Unknow mode!')
            return

        self.name = name
        self.lang_mode = mode
        self.im_load_fn = load_fn

        self.lang = None

        # read caption pairs
        if pairs is not None:
            self.pairs = pairs
        else:
            assert(annFile is not None and image_root is not None)
            self.pairs = utils.read_captions(annFile, image_root)

        self.max_sent_num = None
        self.max_word_num = None

    def stat(self):

        w_max, s_max = 0, 0
        for path, caption in self.pairs:
            s_max = s_max if len(caption) < s_max else len(caption)
            for sent in caption:
                if self.lang_mode == 'word':
                    import fool
                    cur_len = len([term.strip() for term in fool.cut(sent)[0] if len(term.strip())])
                else:
                    assert(self.lang_mode == 'char')
                    cur_len = len([w.strip() for w in sent if len(w.strip()) > 0])

                w_max = w_max if cur_len < w_max else cur_len

        print('\n#{}# statistics'.format(self.name))
        print('Total len: ', self.__len__())
        print("Max number of sentences: ", s_max)
        print("Max length of sentences: ", w_max)
        self.lang.stat()
        print('')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        cap_var, stop_var = self.variable_from_caption(index)
        image_var, seg_var = self.variable_from_image_path(index)
        return image_var, seg_var, cap_var, stop_var

    def split_train_val(self, prop):
        l = len(self.pairs)
        train_pairs = self.pairs[int(l*prop):]
        val_pairs = self.pairs[:int(l*prop)]

        train_ds = Dataset(self.name+'_train', pairs=train_pairs, mode=self.lang_mode, load_fn=self.im_load_fn)
        val_ds = Dataset(self.name + '_val', pairs=val_pairs, mode=self.lang_mode, load_fn=self.im_load_fn)

        train_ds.set_caption_len(self.max_sent_num, self.max_word_num)
        val_ds.set_caption_len(self.max_sent_num, self.max_word_num)
        return train_ds, val_ds

    def display(self):
        for a in dir(self):
            if not callable(getattr(self, a)) and not a.startswith("__"):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n\n")

    def set_caption_len(self, max_sent_num, max_word_num):
        self.max_sent_num = max_sent_num
        self.max_word_num = max_word_num

    def shuffle(self):
        import random
        random.shuffle(self.pairs)

    def variable_from_caption(self, index):
        cap = self.pairs[index][1]
        indices = []
        for sent in cap:
            indices.append([self.lang.word2idx[word.strip()] for word in sent if len(word.strip()) > 0])
        stop = [0 if i < len(indices) else 1 for i in range(self.max_sent_num)]

        max_len = max([len(sent) for sent in indices])
        # append End_Of_Sequence token; increase to the same size
        for sent in indices:
            sent.extend([EOS_INDEX] * (max_len - len(sent) + 1))
        indices = torch.autograd.Variable(torch.LongTensor(indices)).view(-1, max_len + 1)
        stop = torch.autograd.Variable(torch.LongTensor(stop)).view(-1, 1)
        return indices.cuda() if torch.cuda.is_available() else indices, stop.cuda() \
            if torch.cuda.is_available() else stop

    def variable_from_image_path(self, index):
        image_path = self.pairs[index][0]
        im = np.array(self.im_load_fn(image_path))
        # im = T.resize(im, (512, 512, 3), mode='reflect')

        im_data = np.zeros([1, 4, 240, 240])  # IU chest X-Ray (COCO 640x480)
        im_data[0, 0, ...], im_data[0, 1, ...], im_data[0, 2, ...], im_data[0, 3, ...] = \
            im[0, :, :, 75], im[1, :, :,75], im[2, :,:,75], im[3,:,:,75]
        seg_data = np.zeros([1, 240, 240])
        seg_data[0, ...] = im[4, :, :, 75]
        seg_data[seg_data == 4] = 3
        seg_data = seg_data.astype(np.int16)

        im_data = torch.autograd.Variable(torch.FloatTensor(im_data))
        im_data = im_data.cuda() if torch.cuda.is_available() else im_data

        seg_data = torch.autograd.Variable(torch.LongTensor(seg_data))
        seg_data = seg_data.cuda() if torch.cuda.is_available() else seg_data
        return im_data, seg_data