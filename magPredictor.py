import multiprocessing
import math
import time
import threading
import datetime
from queue import Queue
import sys

import numpy as np
import pyqtgraph as pg
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from readData import readSerial, plotB, h, q2m, SLAVES, MOMENT
from dataViewer import magViewer
from trajectoryView import track3D



class MagPredictor():
    stateNum = 7  # x, y, z, q0, q1, q2, q3
   
    def __init__(self):
        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3-self.stateNum)
        self.dt = 0.03  # 时间间隔[s]
        self.ukf = UKF(dim_x=self.stateNum, dim_z=SLAVES*3, dt=self.dt, points=self.points, fx=self.f, hx=h)
        self.ukf.x = np.array([0.0, 0.0, 0.04, 1, 0, 0, 0])  # 初始值
        self.ukf.R *= 3    # 先初始化为3，后面自适应赋值

        self.ukf.P = np.eye(self.stateNum) * 0.016

        self.ukf.Q = np.eye(self.stateNum) * 0.01 * self.dt     # 将速度作为过程噪声来源，Qi = [v*dt]

        for i in range(3, 7):
            self.ukf.Q[i, i] = 0.0001

    def f(self, x, dt):
        A = np.eye(self.stateNum)
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def run(self, magData, state):
        pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        m = q2m(self.ukf.x[3], self.ukf.x[4], self.ukf.x[5], self.ukf.x[6])
        # print(r'pos={}m, vel={}m/s, e_moment={}'.format(pos, vel, m))

        z = np.hstack(magData[:])
        # 自适应 R
        for i in range(SLAVES * 3):
            # 1.sensor的方差随B的关系式为：Bvar =  2*E(-16)*B^4 - 2*E(-27)*B^3 + 2*E(-8)*B^2 + 1*E(-18)*B + 10
            # Bm = magData[i] + magBgDataShare[i]
            # self.ukf.R[i, i] = (2 * math.exp(-16) * Bm ** 4 - 2 * math.exp(-27) * Bm ** 3 + 2 * math.exp(-8) * Bm * Bm + math.exp(-18) * Bm + 10) * 0.005

            # 2.均值平滑，sensor的方差随B的关系式为：Bvar =  1*E(-8)*B^2 - 2*E(-6)*B + 0.84
            Bm = magData[i] + magBgDataShare[i]
            self.ukf.R[i, i] = (math.exp(-8) * Bm ** 2 - 2 * math.exp(-6) * Bm + 0.84) * 1

            # 3.均值平滑，sensor的方差随B的关系式为：Bvar =  1*E(-8)*B^2 + 6*E(-6)*B + 3.221
            # Bm = magData[i] + magBgDataShare[i]
            # self.ukf.R[i, i] = 1 * math.exp(-8) * Bm ** 2 + 6 * math.exp(-6) * Bm + 3.221

        t0 = datetime.datetime.now()
        self.ukf.predict()
        self.ukf.update(z)
        timeCost = (datetime.datetime.now() - t0).total_seconds()

        state[:] = np.concatenate((mp.ukf.x, np.array([MOMENT, timeCost])))  # 输出的结果


def plotError(mp, slavePlot=0):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer")
    win.setWindowTitle("slave {}: residual".format(slavePlot))
    win.resize(1500, 500)
    pg.setConfigOptions(antialias=True)

    px = win.addPlot(title="y")
    px.addLegend()
    px.setLabel('left', 'x', units='1')
    px.setLabel('bottom', 'points', units='1')
    curvex = px.plot(pen='r')

    py = win.addPlot(title="x")
    py.addLegend()
    py.setLabel('left', 'y', units='1')
    py.setLabel('bottom', 'points', units='1')
    curvey = py.plot(pen='g')

    pz = win.addPlot(title="z")
    pz.addLegend()
    pz.setLabel('left', 'z', units='1')
    pz.setLabel('bottom', 'points', units='1')
    curvez = pz.plot(pen='b')

    n, Rx, Ry, Rz, i = Queue(), Queue(), Queue(), Queue(), 0

    def update():
        nonlocal i, slavePlot
        i += 1
        n.put(i)
        Rx.put(mp.ukf.K[2, 14])
        Ry.put(mp.ukf.y[13])
        Rz.put(mp.ukf.y[14])

        if i > 500:
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
    # 多进程之间共享数据
    magOriginDataShare = multiprocessing.Array('f', range(27))
    magBgDataShare = multiprocessing.Array('f', range(27))
    magSmoothData = multiprocessing.Array('f', range(27))
    magPredictData = multiprocessing.Array('f', range(27))
    state = multiprocessing.Array('f', range(9))  #x, y, z, q0, q1, q2, q3, moment, timeCost

    # 读取sensor数据
    pRead = multiprocessing.Process(target=readSerial, args=(magOriginDataShare, magSmoothData))
    pRead.daemon = True
    pRead.start()

    # 启动定位，放置好胶囊
    time.sleep(3)
    input('go on?')
    mp = MagPredictor()

    # 启动mag3D视图
    # pMagViewer = multiprocessing.Process(target=magViewer, args=(state,))
    # pMagViewer.daemon = True
    # pMagViewer.start()

    # 实时显示sensor的值
    # plotBwindow = multiprocessing.Process(target=plotB, args=(magOriginDataShare, (1, 5, 9), state))
    # plotBwindow.daemon = True
    # plotBwindow.start()

    # 显示残差
    # threadplotError = threading.Thread(target=plotError, args=(mp, 0))
    # threadplotError.start()

    # 显示3D轨迹
    trajectory = multiprocessing.Process(target=track3D, args=(state,))
    trajectory.daemon = True
    trajectory.start()

    while True:
        mp.run(magSmoothData, state)
