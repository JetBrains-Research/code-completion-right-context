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
        '-u', '--use_first_n_objects', type=int, default=None,
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
