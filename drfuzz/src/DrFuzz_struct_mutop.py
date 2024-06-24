import random

import src.data_transforms as Data_transforms
import numpy as np

def gaussian_noise(tc, dataset, params):
    parameters_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    result = Data_transforms.gaussian_noise(tc, random.choice(parameters_list), params, dataset)
    return result

def uniform_noise(tc, dataset, params):
    parameters_list = np.linspace(0, 1, 20)
    return Data_transforms.uniform_noise(tc, random.choice(parameters_list), params, dataset)


def multiplicative_noise(tc, dataset, params):
    parameters_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    return Data_transforms.multiplicative_noise(tc, random.choice(parameters_list), params, dataset)

def swap_label(tc, dataset, params):
    return Data_transforms.swap_label(tc, dataset, params)

def get_mutation_ops_name():
    return ['uniform_noise', 'gauss_noise', 'multiplicative_noise', 'swap_label']

def get_mutation_func(name):
    if name == 'gauss_noise':
        return gaussian_noise
    elif name == 'multiplicative_noise':
        return multiplicative_noise
    elif name == "uniform_noise":
        return uniform_noise
    elif name == "swap_label":
        return swap_label