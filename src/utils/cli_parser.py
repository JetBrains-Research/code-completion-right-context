import argparse
import sys

from .technical import load_module


def int_or_none(value):
    if value is None or value.strip().lower() == 'none':
        return None
    return int(value)


def parse_bool(value):
    if value is True or value.strip().lower() == 'true':
        return True
    elif value is False or value.strip().lower() == 'false':
        return False
    raise ValueError('Set True or False for -d [--distributed]')


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
        '-u', '--use_first_n_objects', type=int_or_none, default=None,
        help='Use only first n objects in dataset. Use it for debug.',
    )
    parser.add_argument(
        '-d', '--distributed', type=parse_bool, default=False,
        help='Whether or not to use the distributional training mode.',
    )
    return parser


def parse_arguments(config):
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])
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
