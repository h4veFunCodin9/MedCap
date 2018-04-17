#########################
# Base Configuration
#########################

class Config(object):
    # image embedding size
    IM_EmbeddingSize = 512

    # Size of hidden state in RNN
    SentLSTM_HiddenSize = 512
    WordLSTM_HiddenSize = 512

    # Size of topic vector
    TopicSize = 512

    # Shape of feature map extracted from CNN
    FeatureShape = (512, 16, 16)

    # Training optimizer hyper-parameter
    LR = 1.0e-5
    Momentum = 0.9

    # Path for display example
    TestImagePath = '/mnt/md1/lztao/dataset/IU_Chest_XRay/NLMCXR_png/CXR1100_IM-0068-1001.png'
    '''
    The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. 
    There is no pneumothorax or pleural effusion. There are no focal areas of consolidation. Cholecystectomy clips are present. 
    Small T-spine osteophytes. There is biapical pleural thickening, unchanged from prior. Mildly hyperexpanded lungs.
    '''