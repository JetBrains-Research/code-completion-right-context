import json
import sys
import warnings

from os.path import join as path_join

sys.path.append('../')
warnings.filterwarnings("ignore")

import numpy as np
from catalyst import dl

from src.utils.cli_parser import parse_arguments
from src.modeling.gpt2_config_initializer import GPT2ConfigInitializer
from src.modeling.bi_gpt2_config_initializer import BiGPT2ConfigInitializer

from ddp_models import DDPParameters, DDPSupervisedRunner, create_train_config


if __name__ == '__main__':
    extra_runner_kwargs = {}
    extra_logger_kwargs = {}

    train_config = parse_arguments()

    np.random.seed(train_config.SEED)

    with open(path_join(train_config.HOME_DIR, '_config.json'), 'w') as f:
        json.dump(create_train_config(train_config), f)

    if train_config.TYPE_MODEL == 'GPT2':
        initializer = GPT2ConfigInitializer(config=train_config)
    elif train_config.TYPE_MODEL == 'BiGPT2':
        initializer = BiGPT2ConfigInitializer(config=train_config)
        extra_runner_kwargs['input_key'] = ['input_tensor', 'reverted_input_tensor']
    else:
        raise ValueError(f'Strange model type: \'{train_config.TYPE_MODEL}\'')
    training_parameters = initializer.init_all()

    if train_config.use_distributed_mode:
        runner_initializer = DDPSupervisedRunner
        extra_runner_kwargs['ddp_parameters'] = DDPParameters(
            datasets=training_parameters['datasets'],
            config=train_config,
        )
        extra_logger_kwargs['group'] = 'DDP'
    else:
        runner_initializer = dl.SupervisedRunner

    runner = runner_initializer(**extra_runner_kwargs)

    runner.train(
        model=training_parameters['model'],
        criterion=training_parameters['criterion'],
        optimizer=training_parameters['optimizer'],
        scheduler=training_parameters['scheduler'],
        loaders=training_parameters['loaders'],
        logdir=training_parameters['logdir'],
        num_epochs=train_config.N_EPOCH,
        valid_loader='valid',
        ddp=train_config.use_distributed_mode,
        callbacks=training_parameters['callbacks'],
        verbose=True,
        # amp=False,
        loggers={'wandb': dl.WandbLogger(
            project=train_config.WANDB_GROUP,
            name=train_config.MODEL_NAME,
            config=create_train_config(train_config),
            **extra_logger_kwargs
        )}
    )
