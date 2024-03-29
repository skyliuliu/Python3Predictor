import datetime
import multiprocessing
import time

import numpy as np
import matplotlib.pyplot as plt

from readData import q2m, readSerial
from dataViewer import magViewer
from trajectoryView import track3D

tao = 1e-3
delta = 0.0001
eps_stop = 1e-9
eps_step = 1e-6
eps_residual = 1e-3
residual_memory = []
us = []
poss = []
ems = []

SLAVES = 2
MOMENT = 2169
DISTANCE = 0.02
SENSORLOC = np.array([[0, 0, DISTANCE]]).T
EPM = np.array([[0, 0, 1]]).T

def h(state):
    '''
    以外部大磁体为参考系，得出胶囊内sensor的读数
    :param state: 胶囊的位姿状态
    :param EPM: 外部大磁体的朝向
    :return: 两个sensor的读数 (6, )
    '''
    EPMNorm = np.linalg.norm(EPM)
    eEPM = EPM / EPMNorm

    pos, q = state[0: 3], state[3:]
    R = q2R(q)
    d = np.dot(R.T, SENSORLOC * 0.5)  # 胶囊坐标系下的sensor位置矢量转换到EPM坐标系

    r1 = pos.reshape(3, 1) + d
    r2 = pos.reshape(3, 1) - d
    r1Norm = np.linalg.norm(r1)
    r2Norm = np.linalg.norm(r2)
    er1 = r1 / r1Norm
    er2 = r2 / r2Norm

    # 每个sensor的B值[mGs]
    B1 = MOMENT * np.dot(r1Norm ** (-3), np.subtract(3 * np.dot(np.inner(er1, eEPM), er1), eEPM))
    B2 = MOMENT * np.dot(r2Norm ** (-3), np.subtract(3 * np.dot(np.inner(er2, eEPM), er2), eEPM))

    B1s = np.dot(R, B1)
    B2s = np.dot(R, B2)

    return np.vstack((B1s, B2s)).reshape(-1)

def q2R(q):
    '''
    从四元数求旋转矩阵
    :param q: 四元数
    :return: R 旋转矩阵
    '''
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3,     2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3,     1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2,     2 * q2 * q3 + 2 * q0 * q1,     1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])
    return R


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
        delta = 0.0001
    else:
        delta = 0.001
    state1[param_index] += delta
    state2[param_index] -= delta
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


def LM(state2, output_data, n, maxIter=100):
    """
    Levenberg–Marquardt优化算法的主体
    :param state2: 预估的状态量 (n, ) + [moment, costTime]
    :param output_data: 观测量 (m, )
    :param n: 状态量的维度
    :param maxIter: 最大迭代次数
    :return: None
    """
    output_data = np.array(output_data)
    state = np.array(state2)[:n]
    t0 = datetime.datetime.now()
    m = output_data.shape[0]
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
        poss.append(state[:3])
        ems.append(state[3:])
        i += 1
        while True:
            Hessian_LM = A + u * np.eye(n)  # calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)  # calculating the update step
            if np.linalg.norm(step) <= eps_step:
                stateOut(state, state2, t0, i, mse)
                print('threshold_step')
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
                stop = (np.linalg.norm(g, ord=np.inf) <= eps_stop) or (mse <= eps_residual)
                us.append(u)
                residual_memory.append(mse)
                if stop:
                    print('threshold_stop or threshold_residual')
                    stateOut(state, state2, t0, i, mse)
                    return
                else:
                    break
            else:
                u *= v
                v *= 2
                us.append(u)
                residual_memory.append(mse)
        if i == maxIter:
            print('maxIter_step')
            stateOut(state, state2, t0, i, mse)


def stateOut(state, state2, t0, i, mse):
    timeCost = (datetime.datetime.now() - t0).total_seconds()
    state2[:] = np.concatenate((state, np.array([MOMENT, timeCost, i])))  # 输出的结果
    pos = np.round(state[:3], 3)
    R = q2R(state[3:7])
    emx = np.round(R[0, :], 3)
    emz = np.round(R[-1, :], 3)
    print('i={}, pos={}m, emx={}, emz={}, timeCost={:.3f}s, mse={:.8e}'.format(i, pos, emx, emz, timeCost, mse))



def generate_data(num_data, state):
    """
    生成模拟数据
    :param num_data: 数据维度
    :return: 模拟的B值, (num_data, )
    """
    Bmid = h(state)  # 模拟数据的中间值
    std = 10
    Bsim = np.zeros(num_data)

    for j in range(num_data):
        # std = math.sqrt((math.exp(-8) * B[j] ** 2 - 2 * math.exp(-6) * B[j] + 0.84)) * 2
        Bsim[j] = np.random.normal(Bmid[j], std, 1)
    return Bsim


def sim():
    m, n = 6, 7
    state0 = np.array([-0.1, 0.2, -0.5, 1, 0, 0, 0, MOMENT, 0, 0])  # 初始值
    # 真实值
    states = [np.array([0, 0.5, -0.5, 1, 1, 0, 0]),
              np.array([0.2, -0.2, -0.4, 0, 0.7, 0.7, 1])]
    for i in range(1):
        # run
        output_data = generate_data(m, states[i])
        LM(state0, output_data, n, maxIter=150)
        # plot poss and ems
        iters = len(poss)
        for j in range(iters):
            state00 = np.concatenate((poss[j], ems[j]))
            plt.ion()
            plotP(state00, states[i], j * 0.1)
            if j == iters - 1:
                plt.ioff()
                plt.show()
        plotLM(residual_memory, us)
        # residual_memory.clear()
        # us.clear()

def plotLM(residual_memory, us):
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # plt.plot(residual_memory)
    for ax in [ax1, ax2]:
        ax.set_xlabel("iter")
    ax1.set_ylabel("residual")
    ax1.semilogy(residual_memory)
    ax2.set_xlabel("iter")
    ax2.set_ylabel("u")
    ax2.semilogy(us)
    plt.show()

def plotP(state0, state, index):
    pos, q = state0[:3], state0[3:]
    em = q2R(q)[-1, :]
    xtruth = state.copy()[:3]
    xtruth[1] += index  # 获取坐标真实值
    mtruth = q2R(state.copy()[3:])[-1, :]  # 获取姿态真实值
    pos2 = np.zeros(2)
    pos2[0], pos2[1] = pos[1] + index, pos[2]  # 预测的坐标值

    # plt.axis('equal')
    # plt.ylim(0.2, 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(pos2[0], pos2[1], 'b+')
    plt.text(pos2[0], pos2[1], int(index * 10), fontsize=9)
    plt.plot(xtruth[1], xtruth[2], 'ro')  # 画出真实值
    plt.text(xtruth[1], xtruth[2], int(index * 10), fontsize=9)

    # 添加磁矩方向箭头
    scale = 0.05
    plt.annotate(text='', xy=(pos2[0] + em[1] * scale, pos2[1] + em[2] * scale), xytext=(pos2[0], pos2[1]),
                 color="blue", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="b"))
    plt.annotate(text='', xy=(xtruth[1] + mtruth[1] * scale, xtruth[2] + mtruth[2] * scale),
                 xytext=(xtruth[1], xtruth[2]),
                 color="red", weight="bold", arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="r"))
    # 添加坐标轴标识
    plt.xlabel('iter/1')
    plt.ylabel('pos/m')
    # 添加网格线
    plt.gca().grid(b=True)
    # 最大化窗口
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    # 增加固定时间间隔
    plt.pause(0.05)

def main():
    # 多进程之间共享数据
    B0 = multiprocessing.Array('f', range(27))
    Bg = multiprocessing.Array('f', range(27))
    Bs = multiprocessing.Array('f', range(27))
    Bpre = multiprocessing.Array('f', range(18))
    # x, y, z, q0, q1, q2, q3, moment, costTime, iter
    state = multiprocessing.Array('f', [0, 0, 0.04, 1, 0.001, 0, 0, 0.3, 0, 1])

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
        LM(state, Bs, 7)


if __name__ == '__main__':
    sim()
