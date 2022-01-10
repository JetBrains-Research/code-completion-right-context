import sys
import warnings

sys.path.append('../')
warnings.filterwarnings("ignore")

import numpy as np
from catalyst import dl

from src.utils.cli_parser import parse_arguments
from src.modeling.gpt2_config_initializer import GPT2ConfigInitializer

from gpt_config import Config
from ddp_models import DDPParameters, DDPSupervisedRunner

if __name__ == '__main__':
    extra_runner_train_parameters = {}
    extra_runner_parameters = {}

    Config = parse_arguments(Config)

    np.random.seed(Config.SEED)

    initializer = GPT2ConfigInitializer(config=Config)
    training_parameters = initializer.init_all()
    config = {
        'model_name': Config.MODEL_NAME,
        'sequence_length': Config.SEQUENCE_LENGTH,
        'tokenizer': Config.TOKENIZER,
        'num_epochs': Config.N_EPOCH,
        'batch_size': Config.BATCH_SIZE,
        'h_size': Config.HEAD_SIZE,
        'n_heads': Config.N_HEADS,
        'vocab_size': Config.VOCAB_SIZE,
        'n_layers': Config.N_LAYERS,
        'dropout': Config.DROPOUT,
        'optim': (
            Config.OPTIMIZER_NAME if hasattr(Config, 'OPTIMIZER_NAME')
            else str(Config.OPTIMIZER_CLASS)
        ),
        **{
            f'optim_{key}': f'{value}'
            for key, value in Config.OPTIMIZER_ADDITIONAL_ARGUMENTS.items()
        },
        'max_norm': Config.MAX_NORM,
        'accumulation_steps': Config.ACCUMULATION_STEPS,
        'scheduler': (
            Config.SCHEDULER_NAME if hasattr(Config, 'SCHEDULER_NAME')
            else str(Config.SCHEDULER_CLASS)
        ),
        **{
            f'scheduler_{key}': f'{value}'
            for key, value in Config.SCHEDULER_ADDITIONAL_ARGUMENTS.items()
        },
        'distributed_mode': Config.use_distributed_mode,
        'config_name': Config.config_name,
        'logdir': Config.HOME_DIR,
        'num_workers': Config.NUM_WORKERS,
    }

    logger = dl.WandbLogger(project=Config.WANDB_GROUP, name=Config.MODEL_NAME)
    if logger is not None:
        logger.log_hparams(hparams=config)

    if Config.use_distributed_mode:
        runner_initializer = DDPSupervisedRunner
        extra_runner_parameters = {
            'ddp_parameters': DDPParameters(
                datasets=training_parameters['datasets'],
                config=Config,
                loggers={'wandb': logger},
            ),
        }
    else:
        runner_initializer = dl.SupervisedRunner
        extra_runner_train_parameters['loggers'] = {'wandb': logger}

    if Config.TYPE_MODEL == 'BiGPT2':
        extra_runner_parameters['input_key'] = ['input_tensor', 'reverted_input_tensor']

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
        amp=False,
        **extra_runner_train_parameters
    )
