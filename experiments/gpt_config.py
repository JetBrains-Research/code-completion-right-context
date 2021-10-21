import torch.nn as nn
import torch.optim as optim

from functools import partial

class Config:
    # global parameters
    HOME_DIR = '/home/popov'
    DATA_DIR = '/home/popov/data/rcompletion/tokenized_data_1806'
    WANDB_GROUP = 'gpt'
    MODEL_NAME = 'gpt2'
    CHECKPOINT_PATH = None

    # dataset mode
    DATASET_TRAIN_MODE = 'lm' # lm or padding or chunks
    DATASET_VALID_MODE = 'lm' # lm or padding or chunks
    DATASET_ADDITIONAL_ARGUMENTS = dict()

    # tokenizer parameters
    TOKENIZER = 'bert'
    VOCAB_SIZE = 16

    # model parameters
    DROPOUT = 0.1
    N_LAYERS = 4
    N_HEADS = 8
    HEAD_SIZE = 256

    # training constant parameters
    BATCH_SIZE = 128
    SEQUENCE_LENGTH = 512
    N_EPOCH = 10

    # optimization classes
    # must NOT be initialized here!
    CRITERION_CLASS = partial(nn.CrossEntropyLoss, ignore_index=0)
    OPTIMIZER_CLASS = optim.Adam
    OPTIMIZER_ADDITIONAL_ARGUMENTS = {
        'lr': 1e-3,
    }
    ACCUMULATION_STEPS = 2
    MAX_NORM = 10

    SCHEDULER_CLASS = optim.lr_scheduler.OneCycleLR
    SCHEDULER_NAME = 'OneCycleLR'
    SCHEDULER_ADDITIONAL_ARGUMENTS = {
        'max_lr': 1e-3
    }
    SCHEDULER_MODE = None

    # seed
    SEED = 12153