#########################
# Base Configuration
#########################

class Config(object):

   # Segmentation classes
    SegClasses = 4
    # image embedding size
    IM_EmbeddingSize = 512

    # Size of hidden state in RNN
    LSTM_HiddenSize = 512

    # Weight for 2 losses whose sum should be 1
    SegLoss_Weight = 0.5
    CapLoss_Weight = 0.5

    # Shape of feature map extracted from CNN
    FeatureShape = (512, 16, 16)

    # Training optimizer hyper-parameter
    LR = 1.0e-5
    Momentum = 0.9

    # Training procedure
    NumIters = 500
    BatchSize = 16
    PrintFrequency = 1
    PlotFrequency = 1
    SaveFrequency = 50


class BRATSConfig(Config):

    def __init__(self, args):
        super(BRATSConfig, self).__init__()

        self.StoreRoot = args.store_root

        # the maximum number of sentences and maximum number of words per sentence
        self.MAX_WORD_NUM = 180

        # dictionary size
        #self.DICT_SIZE = len(lang)

        # Shape of feature map extracted from CNN
        self.FeatureShape = (1024, 15, 15)

        # Train Configuration
        self.OnlySeg = False

    def display(self):
        print("Configuration: ")
        for a in dir(self):
            if not callable(getattr(self, a)) and not a.startswith("__"):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n\n")
