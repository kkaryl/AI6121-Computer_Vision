"""
@Author: Ong Jia Hui
@GitHub: https://github.com/kkaryl
"""
from __future__ import print_function
import configparser
import io

__all__ = ['load_config_from_file', 'update_config_to_file']

def load_config_from_file(config_path, verbose=True):
    config = configparser.ConfigParser()
    config.read(config_path)

    if verbose:
        print_config(config)

    return config


def print_config(config):
    with io.StringIO() as ss:
        config.write(ss)
        ss.seek(0)
        print(ss.read())

    return config    


def update_config_to_file(config_path, config, verbose=True):
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    if verbose:
        print(f'{config_path} is updated!')
        print('')
        print_config(config)
