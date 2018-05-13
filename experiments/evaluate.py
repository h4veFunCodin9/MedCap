import os
import numpy as np
from . import utils
import torch
from data.language import SOS_INDEX, EOS_INDEX
import random
from pandas import DataFrame


# predict the caption for an image
def evaluate(encoder, decoder, lang, image_var, config):
    input_variable = image_var

    # image representation
    im_embed, pred_seg = encoder(input_variable)

    if config.OnlySeg:
        return pred_seg.data.cpu().numpy()

    # init word decoder by topic vector
    decoder_hidden = im_embed
    decoder_input = torch.autograd.Variable(torch.LongTensor([[SOS_INDEX, ]]))
    decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

    decoded_caption = []
    for word_i in range(config.MAX_WORD_NUM):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output[0].data.topk(1)
        ni = topi[0][0]

        if ni == EOS_INDEX:
            break
        else:
            decoded_caption.append(lang.idx2word[ni])

            decoder_input = torch.autograd.Variable(torch.LongTensor([[ni, ]]))
            decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
    return pred_seg.data.cpu().numpy(), decoded_caption


# randomly choose n images and predict their captions, store the resulted image
def evaluate_pairs(model, lang, dataset, config, n=-1, verbose=False):
    encoder = model['encoder']
    decoder = model['decoder']
    # log roots
    store_path = os.path.join(config.StoreRoot, 'evaluation')
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    seg_store_root = os.path.join(store_path, 'seg')
    if not os.path.exists(seg_store_root):
        os.makedirs(seg_store_root)
    cap_store_path = os.path.join(store_path, 'captions.csv')

    # how many samples to be evaluated
    dataset.shuffle()
    if n < 0:
        n = len(dataset)

    # metrics calculator
    bleu_calculator = utils.BLEUCalculate()
    iou_calculator = utils.IOUCalculate()

    # logs
    pred_captions = {'id': [], 'caption': []}

    for i in range(n):
        if verbose:
            print('{}/{}\r'.format(i, n), end='')

        # prepare data
        im_var, seg_var = dataset.variable_from_image_path(i)
        raw_pair = dataset.pairs[i]
        image_name = os.path.basename(raw_pair[0])
        truth_cap = '。'.join(raw_pair[1])
        truth_seg = seg_var.data.cpu().numpy()

        # evaluate
        pred_seg, pred_cap = None, None
        if config.OnlySeg:
            pred_seg = evaluate(encoder, decoder, lang, im_var, config)
        else:
            pred_seg, pred_cap = evaluate(encoder, decoder, lang, im_var, config)
            pred_cap = '。'.join([''.join(sent) for sent in pred_cap])

        # metrics
        assert(pred_seg is not None)
        iou_calculator.add(truth_seg, pred_seg)
        if pred_cap is not None:
            bleu_calculator.add(truth_cap, pred_cap)

        # store seg results
        np.save(os.path.join(seg_store_root, image_name), pred_seg)

    # store caption results
    if not config.OnlySeg:
        df = DataFrame(pred_captions)
        df.to_csv(cap_store_path)

    return iou_calculator.get_iou(), bleu_calculator.get_scores()


def display_randomly(model, lang, dataset, config):
    encoder = model['encoder']
    decoder = model['decoder']

    i = random.randint(0, len(dataset)-1)
    im_var, seg_var = dataset.variable_from_image_path(i)
    raw_pair = dataset.pairs[i]
    truth_cap = raw_pair[1]
    print("Truth: ", truth_cap)
    seg, pred_cap = evaluate(encoder, decoder, lang, im_var, config)
    print("Prediction:", ''.join(pred_cap))
