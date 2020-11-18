import multiprocessing
import time
from queue import Queue
import sys
import threading

import numpy as np
import pyqtgraph as pg
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from readData import readSerial, plotB
from dataViewer import magViewer


class MagPredictor():
    def __init__(self):
        self.slaves = 9
        self.stateNum = 10  # x, y, z, vx, vy, vz, q0, q1, q2, q3
        self.moment = 0.29    # 胶囊的磁矩[A*m^2]
        self.distance = 0.12 # sensor之间的距离[m]
        self.sensorLoc = np.array([[-self.distance, self.distance, 0], [0, self.distance, 0], [self.distance, self.distance, 0],
                                    [-self.distance, 0, 0], [0, 0, 0], [self.distance, 0, 0],
                                    [-self.distance, -self.distance, 0], [0, -self.distance, 0], [self.distance, -self.distance, 0]])

        self.defaultR = 300
        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3-self.stateNum)
        self.dt = 0.01
        self.ukf = UKF(dim_x=self.stateNum, dim_z=self.slaves*3, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = np.array([-0.03, 0.06, 0.001, 0, 0, 0, 0, 1, 0, 0])  # 初始值
        self.ukf.R = np.diag((100, 100, 200) * self.slaves)
        self.ukf.Q = np.eye(self.stateNum) * 0.005
        # self.ukf.Q[0: 3, 0: 3] = Q_discrete_white_noise(3, dt=0.2, var=2.)
        # self.ukf.Q[3: 7, 3: 7] = Q_discrete_white_noise(4, dt=0.2, var=0.5)
        self.ukf.P *= 50
        self.eps = 0
        self.A = np.eye(self.stateNum)
        for i in range(3):
            self.A[0 + i, i + 3] = self.dt


    def f(self, x, dt):
        return np.hstack(np.dot(self.A, x.reshape(self.stateNum, 1)))

    def h(self, state):
        B = np.zeros((self.slaves, 3))
        x, y, z = state[0: 3]
        q0, q1, q2, q3 = state[6: self.stateNum]
        mNorm = np.array([self.q2m(q0, q1, q2, q3)])
        rotNorm = np.array([self.q2m(q0, q1, q2, q3)] * 9)

        pos = np.array([[x, y, z]] * 9) - self.sensorLoc
        r = np.linalg.norm(pos, axis=1, keepdims=True)
        posNorm = pos / r

        B = self.moment * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm), rotNorm)) # 每个sensor的B值[mGs]
        data = B.reshape(-1)
        # print(data)
        return data

    def q2m(self, q0, q1, q2, q3):
        qq2 = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        mx = 2 * (-q0 * q2 + q1 * q3) / qq2
        my = 2 * (q0 * q1 + q2 * q3) / qq2
        mz = (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) / qq2
        return [round(mx, 2), round(my, 2), round(mz, 2)]

    def run(self, magFilterDataShare):
        pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        vel = (round(self.ukf.x[3], 3), round(self.ukf.x[4], 3), round(self.ukf.x[5], 3))
        m = self.q2m(self.ukf.x[6], self.ukf.x[7], self.ukf.x[8], self.ukf.x[9])
        print(r'pos={}m, vel={}m/s, e_moment={}'.format(pos, vel, m))
        # print(self.ukf.y)

        z = np.hstack(magFilterDataShare[:])
        self.ukf.predict()
        self.ukf.update(z)

def plotError(mp, slavePlot=0):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer - By Liu Liu")
    win.setWindowTitle("slave {}: residual".format(slavePlot))
    win.resize(1500, 500)
    pg.setConfigOptions(antialias=True)

    px = win.addPlot(title="Bx")
    px.addLegend()
    px.setLabel('left', 'B', units='mG')
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
        Rx.put(mp.ukf.y[slavePlot * 3])
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
    magOriginDataShare = multiprocessing.Array('f', range(27))
    magFilterDataShare = multiprocessing.Array('f', range(27))

    processRead = multiprocessing.Process(target=readSerial, args=(magOriginDataShare, magFilterDataShare))
    processRead.daemon = True
    processRead.start()

    # 实时显示sensor的值
    # processPlotB = multiprocessing.Process(target=plotB, args=(magOriginDataShare, magFilterDataShare, 0))
    # processPlotB.daemon = True
    # processPlotB.start()

    # 启动定位，放置好胶囊
    time.sleep(3)
    input('go on?')
    mp = MagPredictor()

    # 启动3D视图
    threadmagViewer = threading.Thread(target=magViewer, args=(mp,))
    # threadmagViewer.daemon = True
    threadmagViewer.start()

    # 显示残差
    # threadplotError = threading.Thread(target=plotError, args=(mp, 0))
    # # threadplotError.daemon = True
    # threadplotError.start()

    while True:
        mp.run(magFilterDataShare)
