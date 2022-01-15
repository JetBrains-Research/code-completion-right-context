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
        '-u', '--use_first_n_objects', type=int_or_none, default=None,
        help='Use only first n objects in dataset. Use it for debug.',
    )
    parser.add_argument(
        '-distributed', action='store_true',
        help='Use the distributional training mode.',
    )
    return parser


def parse_arguments():
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])
    module = load_module(namespace.config)
    config = module.Config
    # save all information to config fields
    config.use_distributed_mode = namespace.distributed
    config.use_first_n_objects = namespace.use_first_n_objects
    config.config_name = namespace.config

    return config
