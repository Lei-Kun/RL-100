import numpy as np
from scipy.spatial.transform import Rotation

def matrix_to_euler(matrix):
    euler = Rotation.from_matrix(matrix).as_euler('xyz', degrees=True)
    return np.array(euler, dtype=np.float32)

def euler_to_matrix(euler):
    matrix = Rotation.from_euler('xyz', euler).as_matrix()
    return np.array(matrix, dtype=np.float32)

def matrix_to_rot6d(matrix):
    return matrix.T.reshape(9)[:6]

def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(np.cross(x, y, axis=-1))
    matrix = np.stack([x, y, z], axis=-2).T
    return matrix

def euler_to_rot6d(euler):
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)

def rot6d_to_euler(rot6d):
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)

def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

import zarr
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def rpy_to_rot6d(rpy):
    # rpy: (..., 3)
    rotmat = R.from_euler('xyz', rpy).as_matrix()  # (..., 3, 3)
    rot6d = rotmat[..., :, :2].reshape(*rpy.shape[:-1], 6)  # (..., 6)
    return rot6d

def rot6d_to_rpy(rot6d):
    rot6d_reshaped = rot6d.reshape(3, 2)
    
    third_col = np.cross(rot6d_reshaped[:, 0], rot6d_reshaped[:, 1])
    third_col = third_col / np.linalg.norm(third_col, axis=-1, keepdims=True)
    
    rotmat = np.concatenate([rot6d_reshaped, third_col[:, None]], axis=-1)
    
    rpy = R.from_matrix(rotmat).as_euler('xyz')
    return rpy


if __name__ == '__main__':
    rot6d = np.array([
        9.9953383e-01, -3.7908240e-03, -4.0289816e-03, -9.9977541e-01, 8.5195545e-03, -1.2190901e-03
    ])
    euler = rot6d_to_rpy(rot6d)
    print(euler)
    