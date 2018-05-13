import argparse
from data.dataset import Dataset
from data import utils as data_utils
import numpy as np
from experiments.config import BRATSConfig
from model import utils as model_utils
from experiments.train import train_iters
from experiments.evaluate import evaluate_pairs
import pickle
import os

parser = argparse.ArgumentParser(description="Medical Captioning")
parser.add_argument('command', default='train')
parser.add_argument('--im', required=False,
                    default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/dataset/BRATS/images',
                    metavar="path/to/image/dataset",
                    help="The image dataset")
parser.add_argument('--trainval-cap', required=False,
                    default="/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/dataset/BRATS/train_captions.txt",
                    metavar='path/to/trainval/findings',
                    help="The medical image captions for training and validation")
parser.add_argument('--test-cap', required=False,
                    default="/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/dataset/BRATS/test_captions.txt",
                    metavar="path/to/test/findings",
                    help='The medical image captions for testing')
parser.add_argument('--store-root', required=False,
                    default='/Users/luzhoutao/courses/毕业论文/IU Chest X-Ray/MedCap/checkpoints',
                    # "/mnt/md1/lztao/models/med_cap",
                    metavar='path/to/store/models',
                    help="Store model")
parser.add_argument('--load-root', required=False, default=None,
                    metavar='path/to/saved/models',
                    help="the path to models for restore.")
parser.add_argument('--val-prop', required=False, default=0.1,
                    metavar='proportionate of validation dataset')
parser.add_argument('--seg-mode', required=False, default='word',
                    metavar='how to conduct word segmentation')
parser.add_argument('--start-iter', required=False, default=1,
                    metavar='the start iteration number')

args = parser.parse_args()
print("Arguments: ")
print('Command: ', args.command)
print("Image Dataset: ", args.im)
print("Caption Dataset (trainval): ", args.trainval_cap)
print("Caption Dataset (test): ", args.test_cap)
print("Validation Proportionate: ", args.val_prop)
print("Store root: ", args.store_root)

# load configuration
config = BRATSConfig(args)
config.display()

# dataset
if args.load_root is not None:
    print('Loading dataset configuration from file.')
    train_dataset, val_dataset, test_dataset = pickle.load(open(os.path.join(args.load_root, 'dataset.pkl'), 'rb'))
else:
    trainval_dataset = Dataset('BRATS', args.trainval_cap, args.im, mode=args.seg_mode,
                           load_fn=np.load)
    trainval_dataset.set_caption_len(config.MAX_WORD_NUM)
    train_dataset, val_dataset = trainval_dataset.split_train_val(args.val_prop)
    lang = data_utils.generate_lang(train_dataset)
    train_dataset.lang = lang
    val_dataset.lang = lang

    test_dataset = Dataset('BRATS_test', args.test_cap, args.im, mode=args.seg_mode,
                       load_fn=np.load)
    test_dataset.set_caption_len(config.MAX_WORD_NUM)
    test_dataset.lang = lang

train_dataset.stat()
val_dataset.stat()
test_dataset.stat()
pickle.dump([train_dataset, val_dataset, test_dataset], open(os.path.join(args.store_root, 'dataset.pkl'), 'wb'))

config.DICT_SIZE = len(train_dataset.lang)

# model
print("Create model...")
model = model_utils.get_model(config)
if args.load_root:
    model = model_utils.load_model(model, args.load_root)

if args.command == 'train':
    print("--------Train--------")
    train_iters(model, train_dataset, val_dataset, config, start_iter=args.start_iter)

if args.command in ['train', 'test']:
    print("--------Test--------")
    evaluate_pairs(model, train_dataset.lang, test_dataset, config, np.load, verbose=True)
