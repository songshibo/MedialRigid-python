import numpy as np
from transformations import euler_matrix, quaternion_matrix, concatenate_matrices, quaternion_from_euler, is_same_transform


def identity():
    return np.eye(4)


def affine(lin, pos):
    lin = np.concatenate([lin, np.zeros((1, 3))], axis=0)
    pos = np.concatenate([pos, np.ones(1)])
    lin = np.concatenate([lin, pos[:, None]], axis=1)
    return lin


def euler_to_quaternion(rotation):
    return quaternion_from_euler(axes='rxyz', *np.radians(rotation))


def translate(pos):
    return affine(np.eye(3), np.array(pos) * np.ones(3))


def rotate_degree(angle):
    return euler_matrix(axes='sxyz', *np.radians(angle))


def rotate_radian(rad):
    return euler_matrix(axes='sxyz', *rad)


def scale(factor):
    return affine(np.eye(3) * np.array(factor), np.zeros(3))


def TRS(_position, _orientation, _scale):
    T = translate(_position)
    R = quaternion_matrix(_orientation)
    S = scale(_scale)
    return concatenate_matrices(T, R, S)
