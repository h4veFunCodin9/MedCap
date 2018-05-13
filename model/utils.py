import torch
import os
from .encoder import Encoder
from .decoder import Decoder

def get_model(config):
    encoder = Encoder(config)
    decoder = None
    if not config.OnlySeg:
        decoder = Decoder(config)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda() if decoder is not None else None
    return {'encoder': encoder, 'decoder': decoder}

def save_model(model, config, suffix=""):
    model_root = os.path.join(config.StoreRoot, 'model')
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # save the model
    print("Saving models...")
    torch.save(model['encoder'].state_dict(), os.path.join(model_root, "encoder_"+suffix))
    if not config.OnlySeg:
        torch.save(model['decoder'].state_dict(), os.path.join(model_root, "decoder_"+suffix))
    print("Done!")


def load_model(model, load_root):
    encoder_path = os.path.join(load_root, 'encoder')
    decoder_path = os.path.join(load_root, 'decoder')

    if os.path.isfile(encoder_path):
        print("Loading the model for 'encoder' from '{}' ...".format(encoder_path))
        model['encoder'].load_state_dict(torch.load(encoder_path))
        print("Loaded!")

    if os.path.isfile(decoder_path):
        print("Loading the model for 'decoder' from '{}' ...".format(decoder_path))
        model['decoder'].load_state_dict(torch.load(decoder_path))
        print("Loaded!")

    return model