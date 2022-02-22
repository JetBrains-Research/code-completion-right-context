import torch.nn as nn
import torch.optim as optim

from functools import partial

from ddp_models import (
    RightEmbeddingConfig,
    RightGPTConfig,
    RightCNNConfig,
    TypeModel,
)


class Config:
    # global parameters
    HOME_DIR = '/content/experiment/EXP'
    # HOME_DIR = '/home/porkhun/model_training/practice/try'
    DATA_DIR = '/content/data'
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
        'SHIFTS': list(range(2, 26))
    }

    # tokenizer parameters
    TOKENIZER = 'bpe'
    VOCAB_SIZE = 16

    # model parameters
    DROPOUT = 0.1
    N_LAYERS = 2
    N_HEADS = 4
    HEAD_SIZE = 128

    RIGHT_MODEL_TYPE = TypeModel.EMB
    STACK_RIGHT_LEFT_FEATURES: bool = True
    ONE_WPE: bool = False
    ONE_WTE: bool = False
    INIT_LM_FROM_WTE: bool = True

    if RIGHT_MODEL_TYPE is TypeModel.GPT2:
        RIGHT_MODEL_CONFIG = RightGPTConfig(
            DROPOUT=0.1,
            HEAD_SIZE=128,
        )
    elif RIGHT_MODEL_TYPE is TypeModel.CNN:
        RIGHT_MODEL_CONFIG = RightCNNConfig()
    else:
        RIGHT_MODEL_CONFIG = RightEmbeddingConfig()

    # training constant parameters
    BATCH_SIZE = 40
    SEQUENCE_LENGTH = 512
    N_EPOCH = 30
    NUM_WORKERS = 1

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
