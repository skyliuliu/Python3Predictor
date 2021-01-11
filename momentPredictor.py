import datetime
import math
import multiprocessing
import time
from queue import Queue
import sys

import numpy as np
import pyqtgraph as pg
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.stats import NESS
from scipy import linalg

from readData import readSerial, plotB, q2m, SLAVES
from dataViewer import magViewer


class MagPredictor:
    def __init__(self):
        self.stateNum = 8  # x,y,z,q0,q1,q2,q3, m
        self.distance = 0.12  # sensor之间的距离[m]
        self.sensorLoc = np.array(
            [[-self.distance, self.distance, 0], [0, self.distance, 0], [self.distance, self.distance, 0],
             [-self.distance, 0, 0], [0, 0, 0], [self.distance, 0, 0],
             [-self.distance, -self.distance, 0], [0, -self.distance, 0], [self.distance, -self.distance, 0]])

        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3 - self.stateNum)
        self.dt = 0.03  # 时间间隔[s]
        self.ukf = UKF(dim_x=self.stateNum, dim_z=SLAVES * 3, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = np.array([0, 0, 0.04, 1, 0, 0, 0, 0.3])  # 初始值
        self.x0 = np.zeros(self.stateNum)  # 初始值
        self.x0[:] = self.ukf.x

        self.ukf.R *= 5  # 先初始化为5，后面自适应赋值

        self.ukf.P = np.eye(self.stateNum) * 0.01
        for i in range(3, 7):
            self.ukf.P[i, i] = 0.001
        self.ukf.P[-1, -1] = 0.01

        self.ukf.Q = np.eye(self.stateNum) * 0.001 * self.dt  # 将速度作为过程噪声来源，Qi = [v*dt]
        for i in range(3, 7):
            self.ukf.Q[i, i] = 0.01
        self.ukf.Q[7, 7] = 0.01

    def f(self, x, dt):
        A = np.eye(self.stateNum)
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def h(self, state):
        B = np.zeros((9, 3))
        x, y, z = state[0:3]
        q0, q1, q2, q3 = state[3:7]
        mNorm = np.array([q2m(q0, q1, q2, q3)])
        rotNorm = np.array([q2m(q0, q1, q2, q3)] * 9)

        pos = np.array([[x, y, z]] * 9) - self.sensorLoc
        r = np.linalg.norm(pos, axis=1, keepdims=True)
        posNorm = pos / r

        B = state[7] * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm),
                                                          rotNorm))  # 每个sensor的B值[mGs]
        data = B.reshape(-1)
        return data

    def run(self, magData, state):
        pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        m = q2m(self.ukf.x[3], self.ukf.x[4], self.ukf.x[5], self.ukf.x[6])
        print(r'pos={}m, e_moment={},moment={:.3f}'.format(pos, m, self.ukf.x[-1]))

        # 自适应 R
        for i in range(SLAVES * 3):
            # 1.sensor的方差随B的关系式为：Bvar =  2*E(-16)*B^4 - 2*E(-27)*B^3 + 2*E(-8)*B^2 + 1*E(-18)*B + 10
            # Bm = magData[i] + Bg[i]
            # self.ukf.R[i, i] = (2 * 10**(-16) * Bm ** 4 - 2 * 10**(-27) * Bm ** 3 + 2 * 10**(-8) * Bm * Bm + 10**(-18) * Bm + 10) * 0.005

            # 2.sensor的方差随B的关系式为：Bvar =  1*E(-8)*B^2 - 2*E(-6)*B + 0.84
            Bm = magData[i] + Bg[i]
            self.ukf.R[i, i] = 10**(-8) * Bm ** 2 - 2 * 10**(-6) * Bm + 0.84

            # 3.sensor的方差随B的关系式为：Bvar =  1*E(-8)*B^2 + 6*E(-6)*B + 3.221
            # Bm = magData[i] + Bg[i]
            # self.ukf.R[i, i] = 10**(-8) * Bm ** 2 + 6 * 10**(-6) * Bm + 3.221

        z = np.hstack(magData[:])

        t0 = datetime.datetime.now()
        self.ukf.predict()
        self.ukf.update(z)
        timeCost = (datetime.datetime.now() - t0).total_seconds()

        state[:] = np.concatenate((self.ukf.x, np.array([timeCost])))  # 输出的结果

        # 计算NEES值
        xtruth = np.array([0.1, 0.1, 0.14, 0, 1, 0, 0, 0.3])
        xes = self.ukf.x
        p = self.ukf.P
        self.nees = np.dot((xtruth - xes).T, linalg.inv(p)).dot(xtruth - xes)
        print('mean NEES is: ', self.nees)


def plotError(mp, slavePlot=4):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer - y")
    win.setWindowTitle("slave {}: residual".format(slavePlot))
    win.resize(1500, 500)
    pg.setConfigOptions(antialias=True)

    px = win.addPlot(title="likelihood")
    px.addLegend()
    px.setLabel('left', 'likelihood', units='1')
    px.setLabel('bottom', 'points', units='1')
    curvex = px.plot(pen='r')

    py = win.addPlot(title="nees")
    py.addLegend()
    py.setLabel('left', 'nees', units='1')
    py.setLabel('bottom', 'points', units='1')
    curvey = py.plot(pen='r')

    pz = win.addPlot(title="Bz")
    pz.addLegend()
    pz.setLabel('left', 'B', units='mG')
    pz.setLabel('bottom', 'points', units='1')
    curvez = pz.plot(pen='r')

    n, Rx, Ry, Rz, i = Queue(), Queue(), Queue(), Queue(), 0

    def update():
        nonlocal i, slavePlot
        i += 1
        n.put(i)
        Rx.put(mp.ukf.likelihood)
        Ry.put(mp.nees)
        Rz.put(mp.ukf.y[slavePlot * 3 + 2])

        if i > 100:
            for q in [n, Rx, Ry, Rz]:
                q.get()

        for (curve, Bqueue) in [(curvex, Rx), (curvey, Ry), (curvez, Rz)]:
            curve.setData(n.queue, Bqueue.queue)

    timer = pg.Qt.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)

    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    # 开启多进程读取数据
    B0 = multiprocessing.Array('f', range(27))
    Bs = multiprocessing.Array('f', range(27))
    Bg = multiprocessing.Array('f', range(27))
    Bpre = multiprocessing.Array('f', range(27))
    state = multiprocessing.Array('f', range(9))  # x, y, z, q0, q1, q2, q3, moment, timeCost

    # processRead = multiprocessing.Process(target=readSerial, args=(B0, Bs, Bg))
    # processRead.daemon = True
    # processRead.start()

    # 启动定位，放置好胶囊
    time.sleep(1)
    input('go on?')
    mp = MagPredictor()

    # 启动mag3D视图
    pMagViewer = multiprocessing.Process(target=magViewer, args=(state,))
    # pMagViewer.daemon = True
    pMagViewer.start()

    # 实时显示sensor的值
    # plotBwindow = multiprocessing.Process(target=plotB, args=(B0, (1, 5, 9), state))
    # plotBwindow.daemon = True
    # plotBwindow.start()

    # 显示残差
    # plotywindow = threading.Thread(target=plotError, args=(mp, ))
    # # plotywindow.daemon = True
    # plotywindow.start()

    # 1、使用UKF预测磁矩
    # while True:
    #     mp.run(Bs, state)

    # 2、直接计算磁偶极矩产生的B值，与测试结果做对比，来校对磁矩值
    # B = mp.h([0, 0, 0.0405, 0.30])
    # for slave in range(9):
    #     Bx = B[slave * 3]
    #     By = B[slave * 3 + 1]
    #     Bz = B[slave * 3 + 2]
    #     print('slave {}: {}'.format(slave + 1, (round(Bx, 2), round(By, 2), round(Bz, 2))))

    # 3、使用模拟的实测结果，测试UKF滤波器的参数设置是否合理
    B = mp.h([0.1, 0.1, 0.14, 0, 1, 0, 0, 0.3])  # 模拟数据的中间值
    n = 100  # 数据个数
    std = 2
    Bsim = np.zeros((27, n))

    for j in range(27):
        # std = math.sqrt((math.exp(-8) * B[j] ** 2 - 2 * math.exp(-6) * B[j] + 0.84)) * 2
        Bsim[j, :] = np.random.normal(B[j], std, n)

    for i in range(n):
        print('=========={}=========='.format(i))
        mp.run(Bsim[:, i], state)
        time.sleep(0.5)
