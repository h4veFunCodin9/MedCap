import torch
import os
from .encoder import Encoder
from .decoder import SentDecoder, WordDecoder

def get_model(config):
    encoder = Encoder(config)
    sent_decoder, word_decoder = None, None
    if not config.OnlySeg:
        sent_decoder = SentDecoder(config)
        word_decoder = WordDecoder(config)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        sent_decoder = sent_decoder.cuda() if sent_decoder is not None else None
        word_decoder = word_decoder.cuda() if word_decoder is not None else None
    return {'encoder': encoder, 'sent_decoder': sent_decoder, 'word_decoder': word_decoder}

def save_model(model, config, suffix=""):
    model_root = os.path.join(config.StoreRoot, 'model')
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # save the model
    print("Saving models...")
    torch.save(model['encoder'].state_dict(), os.path.join(model_root, "encoder_"+suffix))
    if not config.OnlySeg:
        torch.save(model['sent_decoder'].state_dict(), os.path.join(model_root, "sent_decoder_"+suffix))
        torch.save(model['word_decoder'].state_dict(), os.path.join(model_root, "word_decoder_"+suffix))
    print("Done!")


def load_model(model, load_root):
    encoder_path = os.path.join(load_root, 'encoder')
    sent_decoder_path = os.path.join(load_root, 'sent_decoder')
    word_decoder_path = os.path.join(load_root, 'word_decoder')

    if os.path.isfile(encoder_path):
        print("Loading the model for 'encoder' from '{}' ...".format(encoder_path))
        model['encoder'].load_state_dict(torch.load(encoder_path))
        print("Loaded!")

    if os.path.isfile(sent_decoder_path):
        print("Loading the model for 'sent_decoder' from '{}' ...".format(sent_decoder_path))
        model['sent_decoder'].load_state_dict(torch.load(sent_decoder_path))
        print("Loaded!")

    if os.path.isfile(word_decoder_path):
        print("Loading the model for 'word_decoder' from '{}' ...".format(word_decoder_path))
        model['word_decoder'].load_state_dict(torch.load(word_decoder_path))
        print("Loaded!")

    return model