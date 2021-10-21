import argparse
import sys

from .technical import load_module


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help='Path to the config file.',
    )
    parser.add_argument(
        '-n', '--name',
        help='Model name. Name will be used for log name and wandb name.',
    )
    parser.add_argument(
        '-u', '--use_first_n_objects', type=int,
        help='Use only first n objects in dataset. Use it for debug.',
    )
    parser.add_argument(
        '-d', '--distributed', type=bool, default=True,
        help='Whether or not to use the distributional training mode.',
    )
    return parser


def parse_arguments():
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])
    config = load_module(namespace.config).Config
    model_name = (
        namespace.name
        if namespace.name is not None
        else f'{config.WANDB_GROUP}_{config.WANDB_GROUP}_{namespace.config}'
    )

    # save all information to config fields
    config.use_distributed_mode = namespace.distributed
    config.use_first_n_objects = namespace.use_first_n_objects
    config.model_name = model_name
    config.config_name = namespace.config

    return config