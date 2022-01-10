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
    loggers: Dict[str, dl.ILogger]


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

    def get_loggers(self):
        loggers = OrderedDict([
            ("console", dl.ConsoleLogger()),
        ])
        if self.__ddp_params.loggers is not None:
            loggers.update(self.__ddp_params.loggers)
        return loggers
