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

        w_max = 0
        for path, caption in self.pairs:
            if self.lang_mode == 'word':
                import fool
                fool.load_userdict('data/foolnltk_userdict')
                cur_len = len([term for term in fool.cut(caption)[0]])
            else:
                assert(self.lang_mode == 'char')
                cur_len = len(caption)
            w_max = w_max if cur_len < w_max else cur_len

        print('\n#{}# statistics'.format(self.name))
        print('Total len: ', self.__len__())
        print("Max length of sentences: ", w_max)
        self.lang.stat()
        print('')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        cap_var = self.variable_from_caption(index)
        image_var, seg_var = self.variable_from_image_path(index)
        return image_var, seg_var, cap_var

    def split_train_val(self, prop):
        l = len(self.pairs)
        train_pairs = self.pairs[int(l*prop):]
        val_pairs = self.pairs[:int(l*prop)]

        train_ds = Dataset(self.name+'_train', pairs=train_pairs, mode=self.lang_mode, load_fn=self.im_load_fn)
        val_ds = Dataset(self.name + '_val', pairs=val_pairs, mode=self.lang_mode, load_fn=self.im_load_fn)

        train_ds.set_caption_len(self.max_word_num)
        val_ds.set_caption_len(self.max_word_num)
        return train_ds, val_ds

    def display(self):
        for a in dir(self):
            if not callable(getattr(self, a)) and not a.startswith("__"):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n\n")

    def set_caption_len(self, max_word_num):
        self.max_word_num = max_word_num

    def shuffle(self):
        import random
        random.shuffle(self.pairs)

    def variable_from_caption(self, index):
        cap = self.pairs[index][1]
        if self.lang_mode == 'word':
            import fool
            fool.load_userdict('data/foolnltk_userdict')
            terms = fool.cut(cap)
            terms = terms[0]
        else:
            assert(self.lang_mode == 'char')
            terms = list(cap)
        indices = [self.lang.word2idx[term] for term in terms]

        indices.append(EOS_INDEX)
        indices = torch.autograd.Variable(torch.LongTensor(indices)).view(-1, 1)
        return indices.cuda() if torch.cuda.is_available() else indices

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