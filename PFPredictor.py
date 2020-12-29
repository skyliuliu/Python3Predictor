import datetime
import math
import multiprocessing
import time
import threading
from queue import Queue
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import uniform, randn
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
import scipy.stats

from readData import readSerial, plotB, q2m, SLAVES, SENSORLOC
from dataViewer import magViewer

stateNum = 3

def create_uniform_particles(x_range, y_range, z_range, m_range, N):
    particles = np.empty((N, stateNum))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(z_range[0], z_range[1], size=N)
    # particles[:, 3] = uniform(m_range[0], m_range[1], size=N)
    # particles[:, 4] = uniform(-1, 1, size=N)
    # particles[:, 5] = uniform(-1, 1, size=N)
    # particles[:, 6] = uniform(-1, 1, size=N)
    return particles

def create_gaussian_particles(mean, N):
    particles = np.empty((N, stateNum))
    particles[:, 0] = mean[0] + (randn(N) * 0.1)
    particles[:, 1] = mean[1] + (randn(N) * 0.1)
    particles[:, 2] = mean[2] + (randn(N) * 0.1)
    # particles[:, 3] = mean[3] + (randn(N) * 0.1)
    # particles[:, 4] = mean[4] + (randn(N) * 0.3)
    # particles[:, 5] = mean[5] + (randn(N) * 0.3)
    # particles[:, 6] = mean[6] + (randn(N) * 0.3)
    return particles

def predict(particles, dt=0.03):
    n = len(particles)
    particles[:, 0] += randn(n) * 0.001 * dt  # 位置x的过程误差
    particles[:, 1] += randn(n) * 0.001 * dt  # 位置y的过程误差
    particles[:, 2] += randn(n) * 0.001 * dt  # 位置z的过程误差
    # particles[:, 3] += randn(n) * 0.001  # 磁矩m的过程误差
    # particles[:, 4] += randn(n) * 0.001  # x方向单位矢量ex的过程误差
    # particles[:, 5] += randn(n) * 0.001  # y方向单位矢量ex的过程误差
    # particles[:, 6] += randn(n) * 0.001  # z方向单位矢量ex的过程误差

def update(particles, weights, z, R):

    weights.fill(1.)
    for i,sensor in enumerate(SENSORLOC):
        B_array = h(particles, sensor)
        for j in range(3):
            zi = z[i * 3 + j]
            weights *= scipy.stats.norm(B_array[:, j], R).pdf(z[i * 3 + j])
    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize

def h(particles, sensor):
    '''
    计算n个粒子在对应sensor处的B值
    :param particles: 粒子群，n * stateNum，包含每个粒子状态
    :param sensor: sensor坐标 1*3的数组
    :return: B值数组 n*3
    '''
    n = len(particles)
    B = np.zeros((n, 3))

    for i in range(n):
        # moment = particles[i, 3]
        # eNorm = np.linalg.norm(particles[i, -3:], axis=0, keepdims=True)
        # mNorm = particles[i, -3:] / eNorm  # 磁矩方向矢量归一化
        moment = 0.3
        mNorm = np.array([0, 0, 1])

        pos = np.array(particles[i, 0: 3]) - sensor
        r = np.linalg.norm(pos, axis=0, keepdims=True)
        posNorm = pos / r  # 位置矢量归一化

        B[i, :] = moment * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm), mNorm))
    # data = B.reshape(-1)
    return B

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    mean = np.average(particles, weights=weights, axis=0)
    var = np.average((particles - mean)**2, weights=weights, axis=0)
    # x, y, z, m, ex, ey, ez = mean[:]
    x, y, z = mean[:]
    m, ex, ey, ez = 0.3, 0, 0, 1
    eNorm = math.sqrt(ex * ex + ey * ey + ez * ez)
    ex0, ey0, ez0 = round(ex / eNorm, 3), round(ey / eNorm, 3), round(ez / eNorm, 3)
    print('x={:.2f}cm, y={:.2f}cm, z={:.2f}cm, m={:.3f}A*m^2, e={}'.format(x*100, y*100, z*100, m, (ex0, ey0, ez0)))
    return mean, var

def neff(weights):
    x = 1. / np.sum(np.square(weights))
    return x

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill (1.0 / len(weights))

def run(N, R, init=None):
    if init:
        # 初始状态(x, y, z, m, ex, ey, ez)
        particles = create_gaussian_particles(mean=init, N=N)
    else:
        particles = create_uniform_particles((-0.2, 0.2), (-0.2, 0.2), (0.01, 0.1), (0.1, 0.5), N)
    weights = np.zeros(N)

    for i in range(10):
        predict(particles)
        update(particles, weights, magSmoothData, R)

        particlesX = particles[:, 0]
        particlesY = particles[:, 1]
        particlesZ = particles[:, 2]
        # 绘制粒子的3D分布图
        # ax.scatter(particlesX, particlesY, particlesZ)
        # plt.show()
        plt.scatter(particlesX, particlesY, color='k', marker=',', s=weights)
        # plt.show()

        if neff(weights) < N / 2:
            print('==========systematic_resample=============')
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
        mu, var = estimate(particles, weights)

if __name__ == '__main__':
    magOriginData = multiprocessing.Array('f', range(27))
    magSmoothData = multiprocessing.Array('f', range(27))
    magBgDataShare = multiprocessing.Array('f', range(27))
    magPredictData = multiprocessing.Array('f', range(27))
    state = multiprocessing.Array('f', range(9))  # x, y, z, q0, q1, q2, q3, moment, timeCost

    processRead = multiprocessing.Process(target=readSerial, args=(magOriginData, magSmoothData))
    processRead.daemon = True
    processRead.start()

    # 启动定位，放置好胶囊
    time.sleep(3)
    input('go on?')
    init = (0, 0, 0.04, 0.3, 0, 0, 1)
    run(1000, 3, init=None)

    # 启动mag3D视图
    # pMagViewer = multiprocessing.Process(target=magViewer, args=(state,))
    # pMagViewer.daemon = True
    # pMagViewer.start()