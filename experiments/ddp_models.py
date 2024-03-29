from collections import OrderedDict
from typing import Dict, NamedTuple

from catalyst import dl
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)

from gpt_config import Config


class DDPParameters(NamedTuple):
    datasets: Dict[str, Dataset]
    config: Config


class DDPSupervisedRunner(dl.SupervisedRunner):

    def __init__(self, *args, **kwargs):
        self.__ddp_params: DDPParameters = kwargs.pop('ddp_parameters')
        super().__init__(*args, **kwargs)


    def get_loaders(self, stage: str):
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                self.__ddp_params.datasets['train'],
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=True,
            )
            valid_sampler = DistributedSampler(
                self.__ddp_params.datasets['valid'],
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        train_loader = DataLoader(
            self.__ddp_params.datasets['train'],
            batch_size=self.__ddp_params.config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=self.__ddp_params.config.NUM_WORKERS
        )
        valid_loader = DataLoader(
            self.__ddp_params.datasets['valid'],
            batch_size=self.__ddp_params.config.BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=self.__ddp_params.config.NUM_WORKERS
        )
        return {'train': train_loader, 'valid': valid_loader}


def create_train_config(config: Config):
    return {
        'model_name': config.MODEL_NAME,
        'sequence_length': config.SEQUENCE_LENGTH,
        'tokenizer': config.TOKENIZER,
        'num_epochs': config.N_EPOCH,
        'batch_size': config.BATCH_SIZE,
        'h_size': config.HEAD_SIZE,
        'n_heads': config.N_HEADS,
        'vocab_size': config.VOCAB_SIZE,
        'n_layers': config.N_LAYERS,
        'dropout': config.DROPOUT,
        'optim': (
            config.OPTIMIZER_NAME if hasattr(config, 'OPTIMIZER_NAME')
            else str(config.OPTIMIZER_CLASS)
        ),
        **{
            f'optim_{key}': f'{value}'
            for key, value in config.OPTIMIZER_ADDITIONAL_ARGUMENTS.items()
        },
        'max_norm': config.MAX_NORM,
        'accumulation_steps': config.ACCUMULATION_STEPS,
        'scheduler': (
            config.SCHEDULER_NAME if hasattr(config, 'SCHEDULER_NAME')
            else str(config.SCHEDULER_CLASS)
        ),
        **{
            f'scheduler_{key}': f'{value}'
            for key, value in config.SCHEDULER_ADDITIONAL_ARGUMENTS.items()
        },
        'distributed_mode': config.use_distributed_mode,
        'config_name': config.config_name,
        'logdir': config.HOME_DIR,
        'num_workers': config.NUM_WORKERS,
    }
