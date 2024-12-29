import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt

def generate_grid_points(image_size, grid_size):
    """
    生成均匀分布的网格点坐标
    """
    x = np.linspace(0, image_size[0] - 1, grid_size[0])
    y = np.linspace(0, image_size[1] - 1, grid_size[1])
    xv, yv = np.meshgrid(x, y)
    return np.vstack([xv.ravel(), yv.ravel()]).T

def apply_affine_transform(image, matrix):
    """
    应用放射变换到图像
    """
    rows, cols = image.shape[:2]
    return cv2.warpAffine(image, matrix[:2], (cols, rows))


def inverse_affine_transform(matrix):
    """
    计算放射变换矩阵的逆矩阵
    """
    print("matrix shape: ", matrix.shape)
    inverse_matrix = np.linalg.inv(matrix)
    return inverse_matrix

def apply_inverse_affine_transform(image, matrix):
    """
    应用逆放射变换到图像
    """
    inverse_matrix = inverse_affine_transform(matrix)
    print(inverse_matrix)
    rows, cols = image.shape[:2]
    return cv2.warpAffine(image, inverse_matrix[:2], (cols, rows))

def apply_inverse_affine_transform_to_points(points, matrix):
    """
    应用逆放射变换到点坐标
    """
    inverse_matrix = inverse_affine_transform(matrix)
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])
    original_points = (inverse_matrix @ homogeneous_points.T).T
    return original_points[:, :2]

def estimate_affine_transform(original_points, transformed_points):
    """
    根据原始点和变换后的点估计放射变换矩阵
    """
    assert original_points.shape == transformed_points.shape
    assert original_points.shape[0] >= 3  # 至少需要三个点来计算放射变换

    # 生成方程组的矩阵
    A = []
    B = []

    for (x, y), (x_prime, y_prime) in zip(original_points, transformed_points):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(x_prime)
        B.append(y_prime)
    
    A = np.array(A)
    B = np.array(B)
    
    # 求解线性方程组
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # 构造放射变换矩阵
    matrix = np.array([
        [X[0], X[1], X[2]],
        [X[3], X[4], X[5]],
        [0, 0, 1]
    ])

    return matrix


def estimate_affine_transform_matrix(src_points, dst_points):
    """
    根据原始点（src_points）和目标点（dst_points）计算仿射变换矩阵。
    
    参数：
        src_points (array-like): 原始点 (x, y)，形状为 (n, 2)，n为点的数量
        dst_points (array-like): 变换后的点 (x', y')，形状为 (n, 2)
    
    返回：
        numpy.ndarray: 计算得到的仿射变换矩阵 (3x3)
    """
    assert src_points.shape == dst_points.shape
    n = src_points.shape[0]
    assert n >= 3  # 至少需要3个点

    # 构造方程 A * M = B
    A = np.zeros((2 * n, 6))  # 2*n 个方程
    B = np.zeros(2 * n)       # 2*n 个目标变量
    
    for i in range(n):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]
        
        # 第一组方程 (x -> x')
        A[2 * i, 0] = x
        A[2 * i, 1] = y
        A[2 * i, 2] = 1
        A[2 * i, 3] = 0
        A[2 * i, 4] = 0
        A[2 * i, 5] = 0
        B[2 * i] = x_prime
        
        # 第二组方程 (y -> y')
        A[2 * i + 1, 0] = 0
        A[2 * i + 1, 1] = 0
        A[2 * i + 1, 2] = 0
        A[2 * i + 1, 3] = x
        A[2 * i + 1, 4] = y
        A[2 * i + 1, 5] = 1
        B[2 * i + 1] = y_prime
    
    # 使用最小二乘法求解
    # 求解 A * [a, b, c, d, tx, ty] = B
    M = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # 构造仿射变换矩阵
    affine_matrix = np.array([
        [M[0], M[1], M[2]],
        [M[3], M[4], M[5]],
        [0, 0, 1]
    ])
    
    return affine_matrix


def apply_affine_transform_to_points(points, matrix):
    """
    应用放射变换到点坐标
    """
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])
    transformed_points = (matrix @ homogeneous_points.T).T
    return transformed_points[:, :2]