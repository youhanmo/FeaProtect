import random
import numpy as np
from utils.struct_util import get_low_high

def find_n_smallest_indices(lst, n):
    lst_no_zeros = [x for x in lst if x != 0]
    smallest_indices = sorted(range(len(lst_no_zeros)), key=lambda i: lst_no_zeros[i])[:n]
    result = []
    for i in smallest_indices:
        for j in range(len(lst)):
            if lst[j] == lst_no_zeros[i]:
                result.append(j)
                break
    return result

def choose_col(tc, setting, dataset):
    if len(tc.col_select) == 0:
        col_contribution = np.array(tc.col_contribution)
        u_col = np.argsort(col_contribution)[-setting.update_col_num:]
        tc.col_select = u_col
    else:
        u_col = tc.col_select
    return u_col

def gaussian_noise(tc, params, setting, dataset):
    x = tc.input
    u_col = choose_col(tc, setting, dataset)
    u_col = random.choice(u_col)
    mean = 0
    var = params
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, x.shape)
    gauss[~np.in1d(np.arange(gauss.shape[0]), u_col)] = 0
    noisy = x + gauss
    return noisy.astype(np.float32)

def uniform_noise(tc, params, setting, dataset):
    x = tc.input
    u_col = choose_col(tc, setting, dataset)
    u_col = random.choice(u_col)
    noise = np.random.uniform(-params, params, x.shape)
    noise[~np.in1d(np.arange(noise.shape[0]), u_col)] = 0
    noisy = x + noise
    return noisy.astype(np.float32)

def multiplicative_noise(tc, params, setting, dataset):
    x = tc.input
    u_col = choose_col(tc, setting, dataset)
    u_col = random.choice(u_col)
    mean = 0
    var = params
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, x.shape)
    gauss[~np.in1d(np.arange(gauss.shape[0]), u_col)] = 0
    noisy = x + x * gauss
    return noisy.astype(np.float32)

def swap_label(tc, dataset, setting):
    x = tc.input
    out = x.copy()
    u_col = choose_col(tc, setting, dataset)
    low, high, is_enum, enum_value, discrete = get_low_high(dataset['train_df'], setting.dataset)
    col2idx = {str(item): idx for idx, item in enumerate(dataset['v2_columns'])}
    idx2col = {idx: str(item) for idx, item in enumerate(dataset['v2_columns'])}
    col_idx = [col2idx[en] for en in enum_value.keys()]
    u_col = [col for col in u_col if col in col_idx]
    if u_col == []:
        return out
    u_col = random.choice(u_col)
    out[u_col] = random.choice(enum_value[idx2col[u_col]])
    return out