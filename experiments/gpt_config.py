import torch.nn as nn
import torch.optim as optim

from functools import partial

class Config:
    # global parameters
    HOME_DIR = '/home/porkhun/model_training/practice/bi_gpt/try_run'
#     DATA_DIR = '/mnt/data/porkhun/practice/preprocessed_data'
    DATA_DIR = '/mnt/data/porkhun/data/preprocessed_rmd_data/for_ddp_tryes'
    WANDB_GROUP = 'gpt'
    MODEL_NAME = 'gpt2'
    TYPE_MODEL = 'BiGPT2'  # 'GPT2' or 'BiGPT2'
    CHECKPOINT_PATH = None

    # dataset mode
    DATASET_TRAIN_MODE = 'lm' # lm or padding or chunks
    DATASET_VALID_MODE = 'lm' # lm or padding or chunks
    DATASET_ADDITIONAL_ARGUMENTS = dict()

    # tokenizer parameters
    TOKENIZER = 'bpe'
    VOCAB_SIZE = 16

    # model parameters
    DROPOUT = 0.1
    N_LAYERS = 2
    N_HEADS = 4
    HEAD_SIZE = 128

    # training constant parameters
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 512
    N_EPOCH = 1
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