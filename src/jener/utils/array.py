import random

import numpy as np


def random_partition(array, size=100):
    indexes = list(range(len(array)))
    random.shuffle(indexes)
    remain_indexes = sorted(indexes[size:])
    select_indexes = sorted(indexes[:size])

    remain_array = [array[idx] for idx in remain_indexes]
    select_array = [array[idx] for idx in select_indexes]

    return remain_array, select_array


def random_split(array, n=10):
    indexes = list(range(len(array)))
    random.shuffle(indexes)

    arrays = []
    for select_indexes in np.array_split(indexes, n):
        select_indexes.tolist().sort()
        arrays.append([array[idx] for idx in select_indexes])
    return arrays


def slice(length, window, dup=0):
    if length <= window:
        return [(0, length)]
    return [
        (i, min(i + window, length)) for i in list(range(0, length - dup, window - dup))
    ]


def padding(array, seq_len, pad):
    if len(array) >= seq_len:
        return array[:seq_len]

    return array + [pad] * (seq_len - len(array))


def flatten(arrays):
    array = []
    for i in range(len(arrays)):
        array += arrays[i]
    return array
