import datetime
import math
import multiprocessing
import time
import threading
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


class MagPredictor():
    def __init__(self):
        self.stateNum = 8  #x,y,z,q0,q1,q2,q3, m
        self.distance = 0.12  # sensor之间的距离[m]
        self.sensorLoc = np.array([[-self.distance, self.distance, 0], [0, self.distance, 0], [self.distance, self.distance, 0],
                                    [-self.distance, 0, 0], [0, 0, 0], [self.distance, 0, 0],
                                    [-self.distance, -self.distance, 0], [0, -self.distance, 0], [self.distance, -self.distance, 0]])

        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3-self.stateNum)
        self.dt = 0.03  # 时间间隔[s]
        self.ukf = UKF(dim_x=self.stateNum, dim_z=SLAVES*3, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = np.array([0, 0, 0.0415, 1, 0, 0, 0, 0.39])  # 初始值
        self.x0 = np.array([0, 0, 0.0415, 1, 0, 0, 0, 0.29])  # 初始值
        self.ukf.R = np.ones((SLAVES * 3, SLAVES * 3)) * 5    # 先初始化为5，后面自适应赋值
        self.ukf.P = np.eye(self.stateNum) * 0.01

        self.ukf.Q = np.eye(self.stateNum) * 0.01 * self.dt * self.dt     # 将速度作为过程噪声来源，Qi = [v*dt^2]
        for i in range(3, 8):
            self.ukf.Q[i, i] = 0.01

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
            # sensor的方差随B的关系式为：Bvar =  2*E(-16*B^4) - 2*E(-27*B^3) + 2*E(-8*B^2) + 1*E(-18*B) + 10
            Bm = magData[i] + magBgDataShare[i]
            self.ukf.R[i, i] = 2 * math.exp(-16) * Bm ** 4 - 2 * math.exp(-27) * Bm ** 3 + 2 * math.exp(
                -8) * Bm * Bm + math.exp(-18) * Bm + 10

        z = np.hstack(magData[:])

        t0 = datetime.datetime.now()
        self.ukf.predict()
        self.ukf.update(z)
        timeCost = (datetime.datetime.now() - t0).total_seconds()

        state[:] = np.concatenate((self.ukf.x, np.array([timeCost])))  # 输出的结果

        # 计算NEES值，但效果不太好，偏大
        xtruth = self.x0
        xes = self.ukf.x
        p = self.ukf.P
        nees = np.dot((xtruth - xes).T, linalg.inv(p)).dot(xtruth - xes)
        # print('mean NEES is: ', nees)


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

    py = win.addPlot(title="By")
    py.addLegend()
    py.setLabel('left', 'B', units='mG')
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
        Ry.put(mp.ukf.y[slavePlot * 3 + 1])
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
    mp = MagPredictor()

    # 启动mag3D视图
    pMagViewer = multiprocessing.Process(target=magViewer, args=(state,))
    pMagViewer.daemon = True
    pMagViewer.start()

    # 实时显示sensor的值
    # plotBwindow = multiprocessing.Process(target=plotB, args=(magOriginData, (1, 5, 9), state))
    # plotBwindow.daemon = True
    # plotBwindow.start()

    # 显示残差
    # plotywindow = threading.Thread(target=plotError, args=(mp, ))
    # # plotywindow.daemon = True
    # plotywindow.start()

    # 1、使用UKF预测磁矩
    while True:
        mp.run(magSmoothData, state)

    # 2、直接计算磁偶极矩产生的B值，与测试结果做对比，来校对磁矩值
    # B = mp.h([0, 0, 0.0405, 0.30])
    # for slave in range(9):
    #     Bx = B[slave * 3]
    #     By = B[slave * 3 + 1]
    #     Bz = B[slave * 3 + 2]
    #     print('slave {}: {}'.format(slave + 1, (round(Bx, 2), round(By, 2), round(Bz, 2))))
