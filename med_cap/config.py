#########################
# Base Configuration
#########################

class Config(object):
    # image embedding size
    IM_EmbeddingSize = 512

    # Size of hidden state in RNN
    SentLSTM_HiddenSize = 512
    WordLSTM_HiddenSize = 512

    # Weight for 2 losses whose sum should be 1
    StopLoss_Weight = 0.5
    CapLoss_Weight = 0.5

    # Size of topic vector (should be same as the word LSTM hidden size)
    TopicSize = 512

    # Shape of feature map extracted from CNN
    FeatureShape = (512, 16, 16)

    # Training optimizer hyper-parameter
    LR = 1.0e-6
    Momentum = 0.9

    # Training procedure
    NumIters = 500
    BatchSize = 1
    PrintFrequency = 1
    PlotFrequency = 1

    # Path for display example
    TestImagePath = '/mnt/md1/lztao/dataset/IU_Chest_XRay/NLMCXR_png/CXR1100_IM-0068-1001.png'
    '''
    The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. 
    There is no pneumothorax or pleural effusion. There are no focal areas of consolidation. Cholecystectomy clips are present. 
    Small T-spine osteophytes. There is biapical pleural thickening, unchanged from prior. Mildly hyperexpanded lungs.
    '''

    def __init__(self):
        super(Config, self).__init__()

    def display(self):
        print("Configuration: ")
        for a in dir(self):
            if not callable(getattr(self, a)) and not a.startswith("__"):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n\n")