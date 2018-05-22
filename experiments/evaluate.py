import os
import numpy as np
from . import utils
import torch
from data.language import SOS_INDEX, EOS_INDEX
import random
from pandas import DataFrame

# predict the caption for an image
def evaluate(encoder, word_decoder, lang, image_var, config, im_load_fn):
    input_variable= image_var

    # image representation
    im_embed = encoder.feature(input_variable)

    # generate sentence topics and stop distribution
    '''sent_decoder_hidden = sent_decoder.init_hidden()
    sent_decoder_input = im_embed
    topics = []
    for sent_i in range(config.MAX_SENT_NUM):
        sent_decoder_topic, sent_decoder_stop, sent_decoder_hidden = sent_decoder(sent_decoder_input,
                                                                                  sent_decoder_hidden)
        # if it should stop in this step
        topv, topi = sent_decoder_stop.data.topk(1)
        ni = topi[0][0][0]
        if ni == 1:
            break

        topics.append(sent_decoder_topic)
    '''
    # generate each sentence
    decoded_caption = []
    # init word decoder by topic vector
    word_decoder_hidden = im_embed.view(1, 1, -1)
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
def evaluate_pairs(model, lang, dataset, config, im_load_fn, n=-1, iter=1, verbose=False):
    encoder = model['encoder']
    #sent_decoder = model['sent_decoder']
    word_decoder = model['word_decoder']
    # log roots
    store_path = os.path.join(config.StoreRoot, 'evaluation')
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    cap_store_root = os.path.join(store_path, 'captions')
    if not os.path.exists(cap_store_root):
        os.makedirs(cap_store_root)
    cap_store_path = os.path.join(cap_store_root, str(iter)+'.json')

    # how many samples to be evaluated
    dataset.shuffle()
    if n < 0:
        n = len(dataset)

    # metrics calculator
    bleu_calculator = utils.BLEUCalculate()

    # logs
    pred_captions = []

    for i in range(n):
        if verbose:
            print('{}/{}\r'.format(i, n), end='')

        # prepare data
        im_var, seg_var = dataset.variable_from_image_path(i)
        raw_pair = dataset.pairs[i]
        truth_cap = raw_pair[1]
        #image_name = os.path.basename(raw_pair[0])

        # evaluate
        pred_cap = evaluate(encoder, word_decoder, lang, im_var, config, im_load_fn=im_load_fn)

        # metrics
        bleu_calculator.add(truth_cap, pred_cap)

        # add cur result
        pred_captions.append(
            {
                'image_id': os.path.basename(raw_pair[0])[:-4],
                'caption': ' '.join(pred_cap[0])
            }
        )
    # store caption results
    import json
    with open(cap_store_path, 'w+') as f:
        f.write(json.dumps(pred_captions))

    return bleu_calculator.get_scores()


def display_randomly(model, lang, dataset, config, im_load_fn):
    encoder = model['encoder']
    #sent_decoder = model['sent_decoder']
    word_decoder = model['word_decoder']

    i = random.randint(0, len(dataset)-1)
    im_var, seg_var = dataset.variable_from_image_path(i)
    raw_pair = dataset.pairs[i]
    truth_cap = ' '.join([' '.join(sent) for sent in raw_pair[1]])
    print("Truth: ", truth_cap)
    pred_cap = evaluate(encoder, word_decoder, lang, im_var, config, im_load_fn=im_load_fn)
    print("Prediction:", ' '.join(pred_cap[0]))