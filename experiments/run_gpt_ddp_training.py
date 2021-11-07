import sys
sys.path.append('../')

import numpy as np
from catalyst import dl

from src.utils.cli_parser import parse_arguments
from src.modeling.gpt2_config_initializer import GPT2ConfigInitializer


if __name__ == '__main__':
    extra_runner_parameters = {}
    Config = parse_arguments()
    np.random.seed(Config.SEED)

    initializer = GPT2ConfigInitializer(config=Config)
    training_parameters = initializer.init_all()

    if Config.use_distributed_mode:
        extra_runner_parameters['amp'] = False 
    
    if Config.TYPE_MODEL == 'BiGPT2':
        runner = dl.SupervisedRunner(input_key=["input_tensor", "reverted_input_tensor"])
    else:
        runner = dl.SupervisedRunner()
    
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
        **extra_runner_parameters
    )
