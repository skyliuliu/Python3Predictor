import datetime
import multiprocessing
import time

import numpy as np
import matplotlib.pyplot as plt

from readData import q2m, h, SLAVES, MOMENT, readSerial
from dataViewer import magViewer
from trajectoryView import track3D


tao = 1e-3
delta = 0.0001
threshold_stop = 1e-15
threshold_step = 1e-15
threshold_residual = 1e-6
residual_memory = []
us = []


def derive(state, param_index):
    """
    指定状态量的偏导数
    :param state: 预估的状态量 (n, )
    :param param_index: 第几个状态量
    :return: 偏导数 (m, )
    """
    state1 = state.copy()
    state2 = state.copy()
    if param_index < 3:
        state1[param_index] += 0.0003
        state2[param_index] -= 0.0003
    else:
        state1[param_index] += 0.001
        state2[param_index] -= 0.001
    data_est_output1 = h(state1)
    data_est_output2 = h(state2)
    return 0.5 * (data_est_output1 - data_est_output2) / delta

def jacobian(state, m):
    """
    计算预估状态的雅可比矩阵
    :param state: 预估的状态量 (n, )
    :param m: 观测量的个数 [int]
    :return: J (m, n)
    """
    n = state.shape[0]
    J = np.zeros((m, n))
    for pi in range(0, n):
        J[:, pi] = derive(state, pi)
    return J

def residual(state, output_data):
    """
    计算残差
    :param state: 预估的状态量 (n, )
    :param output_data: 观测量 (m, )
    :return: residual (m, )
    """
    data_est_output = h(state)
    residual = output_data - data_est_output
    return residual

def get_init_u(A, tao):
    """
    确定u的初始值
    :param A: J.T * J (m, m)
    :param tao: 常量
    :return: u [int]
    """
    m = np.shape(A)[0]
    Aii = []
    for i in range(0, m):
        Aii.append(A[i, i])
    u = tao * max(Aii)
    return u

def LM(state2, output_data, maxIter=100):
    """
    Levenberg–Marquardt优化算法的主体
    :param state2: 预估的状态量 (n, ) + [moment, costTime]
    :param output_data: 观测量 (m, )
    :param maxIter: 最大迭代次数
    :return: None
    """
    output_data = np.array(output_data)
    state = np.array(state2)[:7]
    t0 = datetime.datetime.now()
    m = output_data.shape[0]
    n = state.shape[0]
    res = residual(state, output_data)
    J = jacobian(state, m)
    A = J.T.dot(J)
    g = J.T.dot(res)
    u = get_init_u(A, tao)  # set the init u
    # u = 100
    v = 2
    rou = 0
    mse = 0

    for i in range(maxIter):
        i += 1

        Hessian_LM = A + u * np.eye(n)  # calculating Hessian matrix in LM
        step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
        if np.linalg.norm(step) <= threshold_step:
            timeCost = (datetime.datetime.now() - t0).total_seconds()
            state2[:] = np.concatenate((state, np.array([MOMENT, timeCost])))  # 输出的结果
            return
        newState = state + step
        newRes = residual(newState, output_data)
        mse = np.linalg.norm(res) ** 2
        newMse = np.linalg.norm(newRes) ** 2
        rou = (mse - newMse) / (step.T.dot(u * step + g))
        if rou > 0:
            state = newState
            res = newRes
            J = jacobian(state, m)
            A = J.T.dot(J)
            g = J.T.dot(res)
            u *= max(1 / 3, 1 - (2 * rou - 1) ** 3)
            v = 2
        else:
            u *= v
            v *= 2
        us.append(u)
        residual_memory.append(mse)

        pos = np.round(state[:3], 3)
        em = np.round(q2m(*state[3:7]), 3)
        # print('i={}, pos={}, m={}, mse={:.8e}'.format(i, pos, em, mse))
        if abs(newMse - mse) < threshold_residual:
            timeCost = (datetime.datetime.now() - t0).total_seconds()
            state2[:] = np.concatenate((state, np.array([MOMENT, timeCost])))  # 输出的结果
            return

def generate_data(num_data):
    """
    生成模拟数据
    :param num_data: 数据维度
    :return: 模拟的B值, (27, )
    """
    Bmid = h([0, 0.1, 0.2, 1, 1, 1, 0])  # 模拟数据的中间值
    std = 1
    Bsim = np.zeros(num_data)

    for j in range(num_data):
        # std = math.sqrt((math.exp(-8) * B[j] ** 2 - 2 * math.exp(-6) * B[j] + 0.84)) * 2
        Bsim[j] = np.random.normal(Bmid[j], std, 1)
    return Bsim

def sim():
    m, n = 27, 7
    state0 = np.array([0, 0, 0.04, 1, 0, 0, 0])  # 初始值
    output_data = generate_data(m)
    # run
    LM(state0, output_data, maxIter=50)
    print(us)
    # plot residual
    plt.plot(residual_memory)
    plt.xlabel("iter")
    plt.ylabel("residual")
    plt.show()

if __name__ == '__main__':
    # 多进程之间共享数据
    B0 = multiprocessing.Array('f', range(27))
    Bg = multiprocessing.Array('f', range(27))
    Bs = multiprocessing.Array('f', range(27))
    Bpre = multiprocessing.Array('f', range(18))
    state = multiprocessing.Array('f', [0, 0, 0.04, 1, 0.001, 0, 0, 0.3, 0])  # x, y, z, q0, q1, q2, q3, moment, costTime

    # 读取sensor数据
    pRead = multiprocessing.Process(target=readSerial, args=(B0, Bs, Bg))
    pRead.daemon = True
    pRead.start()

    # 启动定位，放置好胶囊
    time.sleep(1)
    input('go on?')

    # 启动mag3D视图
    pMagViewer = multiprocessing.Process(target=magViewer, args=(state,))
    pMagViewer.daemon = True
    pMagViewer.start()

    # 显示3D轨迹
    # trajectory = multiprocessing.Process(target=track3D, args=(state,))
    # trajectory.daemon = True
    # trajectory.start()

    # 开始预测
    while True:
        # print('-------------------------------')
        LM(state, Bs)
