import torch.nn as nn
import torch.optim as optim

from functools import partial


class Config:
    # global parameters
    HOME_DIR = '/home/porkhun/model_training/practice/random_shift/2_20'
    # HOME_DIR = '/home/porkhun/model_training/practice/try'
    DATA_DIR = '/mnt/data/porkhun/practice/preprocessed_data'
    # DATA_DIR = '/mnt/data/porkhun/data/preprocessed_rmd_data/for_ddp_tryes'
    WANDB_GROUP = 'Rcompletion'
    MODEL_NAME = 'BiGPT-128'
    TYPE_MODEL = 'BiGPT2'  # 'GPT2' or 'BiGPT2'
    CHECKPOINT_PATH = None
    MODEL_ADDITIONAL_ARGUMENTS = {}

    # dataset mode
    DATASET_TRAIN_MODE = 'lm'  # lm or padding or chunks
    DATASET_VALID_MODE = 'lm'  # lm or padding or chunks
    DATASET_ADDITIONAL_ARGUMENTS = {
        'SHIFTS': list(range(10, 26))
    }

    # tokenizer parameters
    TOKENIZER = 'bpe'
    VOCAB_SIZE = 16

    # model parameters
    DROPOUT = 0.1
    N_LAYERS = 2
    N_HEADS = 4
    HEAD_SIZE = 128

    RIGHT_DROPOUT: float = 0.2  # default None
    RIGHT_HEAD_SIZE: int = 128  # default None
    STACK: bool = True  # default False
    ONE_WPE: bool = False  # default False
    ONE_WTE: bool = True  # default False
    INIT_LM_FROM_WTE: bool = False  # default False

    # training constant parameters
    BATCH_SIZE = 40
    SEQUENCE_LENGTH = 512
    N_EPOCH = 30
    NUM_WORKERS = 2

    # optimization classes
    # must NOT be initialized here!
    CRITERION_CLASS = partial(nn.CrossEntropyLoss, ignore_index=0)
    OPTIMIZER_CLASS = optim.AdamW
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
