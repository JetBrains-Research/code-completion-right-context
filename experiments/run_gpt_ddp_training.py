import sys
sys.path.append('../')

import numpy as np
from catalyst import dl

from src.utils.cli_parser import parse_arguments
from src.modeling.gpt2_config_initializer import GPT2ConfigInitializer

from collections import OrderedDict

if __name__ == '__main__':
    Config = parse_arguments()
    np.random.seed(Config.SEED)

    initializer = GPT2ConfigInitializer(config=Config)
    training_parameters = initializer.init_all()

    runner = dl.SupervisedRunner(device='cuda')

    # if Config.use_distributed_mode:
    #     runner.train(
    #         model=training_parameters['model'],
    #         criterion=training_parameters['criterion'],
    #         optimizer=training_parameters['optimizer'],
    #         scheduler=training_parameters['scheduler'],
    #         datasets=OrderedDict({
    #             'train': training_parameters['datasets']['train'],
    #             'valid': training_parameters['datasets']['valid'],
    #             'batch_size': training_parameters['loaders']['train'].batch_size,
    #             'num_workers': training_parameters['loaders']['train'].num_workers,
    #             'loaders_params': {
    #                 'train': {
    #                     'collate_fn': training_parameters['loaders']['train'].collate_fn,
    #                     'shuffle': training_parameters['loaders']['train'].shuffle,
    #                 },
    #                 'valud': {
    #                     'collate_fn': training_parameters['loaders']['valid'].collate_fn,
    #                     'shuffle': training_parameters['loaders']['train'].shuffle,
    #                 },
    #             },
    #         }),
    #         logdir=training_parameters['logdir'],
    #         num_epochs=Config.N_EPOCH,
    #         valid_loader='valid',
    #         distributed=Config.use_distributed_mode,
    #         callbacks=training_parameters['callbacks'],
    #     )
    # else:
    runner.train(
        model=training_parameters['model'],
        criterion=training_parameters['criterion'],
        optimizer=training_parameters['optimizer'],
        scheduler=training_parameters['scheduler'],
        loaders=training_parameters['loaders'],
        logdir=training_parameters['logdir'],
        num_epochs=Config.N_EPOCH,
        valid_loader='valid',
        distributed=Config.use_distributed_mode,
        callbacks=training_parameters['callbacks'],
    )
