import configparser as cp
import numpy as np


def read_configs(configfile):
    config = cp.ConfigParser()
    config.read(configfile)
    return config


def read_np_array(config_info, dtype):
    return np.array(config_info.split(), dtype=dtype)
