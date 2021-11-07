import sys
import warnings
sys.path.append('../')
warnings.filterwarnings("ignore")

import numpy as np
from catalyst import dl
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.cli_parser import parse_arguments
from src.modeling.gpt2_config_initializer import GPT2ConfigInitializer
from gpt_config import Config


class CustomSupervisedRunner(dl.SupervisedRunner):
    
    def __init__(self, *args, **kwargs):
        self.__dataset_for_ddp_jb = kwargs.pop('dataset_for_ddp')
        self.__config_for_train = kwargs.pop('config_for_train')
        super().__init__(*args, **kwargs)
        
    def get_loaders(self, stage: str):
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                    self.__dataset_for_ddp_jb['train'],
                    num_replicas=self.engine.world_size,
                    rank=self.engine.rank,
                    shuffle=True,
                )
            valid_sampler = DistributedSampler(
                    self.__dataset_for_ddp_jb['valid'],
                    num_replicas=self.engine.world_size,
                    rank=self.engine.rank,
                    shuffle=False,
                )
        else:
            train_sampler = valid_sampler = None

        train_loader = DataLoader(
            self.__dataset_for_ddp_jb['train'],
            batch_size=self.__config_for_train.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=self.__config_for_train.NUM_WORKERS
        )
        valid_loader = DataLoader(
            self.__dataset_for_ddp_jb['valid'],
            batch_size=self.__config_for_train.BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=self.__config_for_train.NUM_WORKERS
        )
        return {"train": train_loader, "valid": valid_loader}


if __name__ == '__main__':   
    extra_runner_train_parameters = {}
    extra_runner_parameters = {}
    
    Config = parse_arguments(Config)
    
    np.random.seed(Config.SEED)

    initializer = GPT2ConfigInitializer(config=Config)
    training_parameters = initializer.init_all()

    if Config.use_distributed_mode:
        runner_initializer = CustomSupervisedRunner
        extra_runner_parameters['dataset_for_ddp'] = training_parameters['datasets']
        extra_runner_parameters['config_for_train'] = Config
        extra_runner_train_parameters['amp'] = False 
    else:
        runner_initializer = dl.SupervisedRunner
    
    if Config.TYPE_MODEL == 'BiGPT2':
        extra_runner_parameters['input_key'] = ["input_tensor", "reverted_input_tensor"]
        
    runner = runner_initializer(**extra_runner_parameters)
    
    runner.train(
        model=training_parameters['model'],
        criterion=training_parameters['criterion'],
        optimizer=training_parameters['optimizer'],
        scheduler=training_parameters['scheduler'],
        loaders=training_parameters['loaders'],
        logdir=training_parameters['logdir'],
        num_epochs=Config.N_EPOCH,
        valid_loader='valid',
        ddp=Config.use_distributed_mode,
        callbacks=training_parameters['callbacks'],
        verbose=True,
        loggers={"wandb": dl.WandbLogger(project=Config.WANDB_GROUP, name=Config.MODEL_NAME)},
        **extra_runner_train_parameters
    )
