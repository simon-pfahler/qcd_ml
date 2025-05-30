"""
This module provides functions to generate complete sets of paths for a given block size.

See the documentation of the functions for more information.
"""

import itertools
import numpy as np


def get_paths_lexicographic(block_size, _gpt_compat=False):
    """
    For a reference point (1,1,1,1) and a point (x_1 + 1, x_2 + 1, x_3 + 1, x_4 + 1) the path 
    is generated as such:
    .. math::

        H_{-4}^{x_4} H_{-3}^{x_3} H_{-2}^{x_2} H_{-1}^{x_1}

    i.e., the dimension 1 is traversed first.

    """
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = sum([[(mu, -1)] for mu, n in enumerate(position) for _ in range(n)], start=[])
        paths.append(path)
    if _gpt_compat is False:
        return paths
    else:
        return [list(reversed(pth)) for pth in paths]


def get_paths_reverse_lexicographic(block_size, _gpt_compat=False):
    """
    Reverse order of get_paths_lexicographic.
    """
    if _gpt_compat is False:
        return [list(reversed(pth)) for pth in get_paths_lexicographic(block_size)]
    else:
        return get_paths_lexicographic(block_size)


def get_paths_one_step_lexicographic(block_size, _gpt_compat=False):
    """
    For a reference point (1,1,1,1) and a point (x_1 + 1, x_2 + 1, x_3 + 1, x_4 + 1) the path
    is generated as such:
    .. math::

        \\cdots H_{-4}H_{-3}H_{-2}H_{-1}H_{-4}H_{-3}H_{-2}H_{-1}
    """
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = []
        pos = np.array(position)
        while pos.any():
            for mu in range(pos.shape[0]):
                if pos[mu] > 0:
                    path.append((mu, -1))
                    pos[mu] -= 1
                if pos[mu] < 0:
                    path.append((mu, 1))
                    pos[mu] += 1
        if _gpt_compat is False:
            paths.append(path)
        else:
            paths.append(list(reversed(path)))
    return paths


def get_paths_one_step_reverse_lexicographic(block_size, _gpt_compat=False):
    """
    For a reference point (1,1,1,1) and a point (x_1 + 1, x_2 + 1, x_3 + 1, x_4 + 1) the path
    is generated as such:
    .. math::

        \\cdots H_{-1}H_{-2}H_{-3}H_{-4}H_{-1}H_{-2}H_{-3}H_{-4}
    """
    paths = []
    for position in itertools.product(*(range(bs) for bs in block_size)):
        path = []
        pos = np.array(position)
        while pos.any():
            for mu in reversed(range(pos.shape[0])):
                if pos[mu] > 0:
                    path.append((mu, -1))
                    pos[mu] -= 1
                if pos[mu] < 0:
                    path.append((mu, 1))
                    pos[mu] += 1
        if _gpt_compat is False:
            paths.append(path)
        else:
            paths.append(list(reversed(path)))
    return paths
