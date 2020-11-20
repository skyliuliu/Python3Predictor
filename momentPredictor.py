import multiprocessing
import time
from queue import Queue
import sys
import threading

import numpy as np
import pyqtgraph as pg
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from readData import readSerial
from dataViewer import magViewer


class MagPredictor():
    def __init__(self):
        self.slaves = 9
        self.stateNum = 4  #x,y,z, m
        self.distance = 0.12  # sensor之间的距离[m]
        self.sensorLoc = np.array([[-self.distance, self.distance, 0], [0, self.distance, 0], [self.distance, self.distance, 0],
                                    [-self.distance, 0, 0], [0, 0, 0], [self.distance, 0, 0],
                                    [-self.distance, -self.distance, 0], [0, -self.distance, 0], [self.distance, -self.distance, 0]])

        self.points = MerweScaledSigmaPoints(n=self.stateNum, alpha=0.3, beta=2., kappa=3-self.stateNum)
        self.dt = 0.03  # 时间间隔[s]
        self.ukf = UKF(dim_x=self.stateNum, dim_z=self.slaves*3, dt=self.dt, points=self.points, fx=self.f, hx=self.h)
        self.ukf.x = np.array([0, 0, 0.0415, 0.29])  # 初始值
        self.ukf.R = np.diag((100, 100, 200) * self.slaves)
        self.ukf.P *= 10

        self.ukf.Q = np.zeros((self.stateNum, self.stateNum))
        # 将加速度作为过程噪声来源，Qi = [[0.5*dt^4, 0.5*dt^3], [0.5*dt^3, dt^2]]
        # self.ukf.Q[0: 6, 0: 6] = Q_discrete_white_noise(dim=2, dt=self.dt, var=1, block_size=3)
        for i in range(4):
            self.ukf.Q[i, i] = 0.05

    def f(self, x, dt):
        A = np.eye(self.stateNum)
        # for i in range(0, 6, 2):
        #     A[i, i + 1] = dt
        return np.hstack(np.dot(A, x.reshape(self.stateNum, 1)))

    def h(self, state):
        B = np.zeros((self.slaves, 3))
        x, y, z = state[0: 3]
        q0, q1, q2, q3, moment = 1, 0, 0, 0, state[-1]
        mNorm = np.array([self.q2m(q0, q1, q2, q3)])
        rotNorm = np.array([self.q2m(q0, q1, q2, q3)] * 9)

        pos = np.array([[x, y, z]] * 9) - self.sensorLoc
        r = np.linalg.norm(pos, axis=1, keepdims=True)
        posNorm = pos / r

        B = moment * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm), rotNorm)) # 每个sensor的B值[mGs]
        data = B.reshape(-1)
        # print(data)
        return data

    def q2m(self, q0, q1, q2, q3):
        qq2 = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        mx = 2 * (-q0 * q2 + q1 * q3) / qq2
        my = 2 * (q0 * q1 + q2 * q3) / qq2
        mz = (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) / qq2
        return [round(mx, 2), round(my, 2), round(mz, 2)]

    def run(self, magOriginDataShare):
        pos = (round(self.ukf.x[0], 3), round(self.ukf.x[1], 3), round(self.ukf.x[2], 3))
        # vel = (round(self.ukf.x[1], 3), round(self.ukf.x[3], 3), round(self.ukf.x[5], 3))
        # em = self.q2m(self.ukf.x[6], self.ukf.x[7], self.ukf.x[8], self.ukf.x[9])
        print(r'pos={}m, moment={}'.format(pos, self.ukf.x[-1]))
        # print(self.ukf.y)

        z = np.hstack(magOriginDataShare[:])
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

def plotB(magOriginDataShare ,slavePlot=(1, 5, 9)):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer")
    win.resize(1500, 1200)
    win.setWindowTitle("slave {}: origin VS KF".format(slavePlot))
    pg.setConfigOptions(antialias=True)

    n = Queue()
    curves = []  # []
    datas = []   # [s1_Bx_Origin, s1_Bx_Predict, s1_By_Origin, s1_By_Predict, ... ]
    for i in slavePlot:
        for Bi in ['Bx', 'By', 'Bz']:
            p = win.addPlot(title='slave {}--'.format(i) + Bi)
            p.addLegend()
            p.setLabel('left', 'B', units='mG')
            p.setLabel('bottom', 'points', units='1')
            cOrigin = p.plot(pen='r', name='Origin')
            # cPredict = p.plot(pen='g', name='Predict')
            curves.append(cOrigin)
            # curves.append(cPredict)
            datas.append(Queue())    # origin
            # datas.append(Queue())    # Predict
        win.nextRow()
    i = 0
    # n, Bx, Bx2, By2, Bz2, By, Bz, i = Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), 0

    def update():
        nonlocal i
        # magPredictData = mp.h(mp.ukf.x)
        i += 1
        n.put(i)
        for slaveIndex, slave in enumerate(slavePlot):
            for Bindex in range(3):
                datas[slaveIndex * 3 + Bindex].put(magOriginDataShare[(slave-1) * 3 + Bindex])
                # datas[slaveIndex * 6 + Bindex * 2 + 1].put(magPredictData[(slave-1) * 3 + Bindex])
        # Bx.put(magOriginDataShare[slave * 3])
        # By.put(magOriginDataShare[slave * 3 + 1])
        # Bz.put(magOriginDataShare[slave * 3 + 2])
        # Bx2.put(magPredictData[slave * 3])
        # By2.put(magPredictData[slave * 3 + 1])
        # Bz2.put(magPredictData[slave * 3 + 2])

        if i > 100:
            n.get()
            for q in datas:
                q.get()
        for (curve, data) in zip(curves, datas):
            curve.setData(n.queue, data.queue)

    timer = pg.Qt.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)

    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    # 开启多进程读取数据
    magOriginDataShare = multiprocessing.Array('f', range(27))
    magPredictData = multiprocessing.Array('f', range(27))

    processRead = multiprocessing.Process(target=readSerial, args=(magOriginDataShare,))
    processRead.daemon = True
    processRead.start()

    # 启动定位，放置好胶囊
    time.sleep(3)
    input('go on?')
    mp = MagPredictor()

    # 启动mag3D视图
    # threadmagViewer = threading.Thread(target=magViewer, args=(mp,))
    # # threadmagViewer.daemon = True
    # threadmagViewer.start()

    # 实时显示sensor的值
    plotBwindow = threading.Thread(target=plotB, args=(magOriginDataShare ,(1, 5, 9)))
    # plotBwindow.setDaemon(True)
    plotBwindow.start()

    # 显示残差
    # threadplotError = threading.Thread(target=plotError, args=(mp, 0))
    # # threadplotError.daemon = True
    # threadplotError.start()

    # while True:
    #     mp.run(magOriginDataShare)

    B = mp.h([0, 0, 0.0405, 0.30])
    for slave in range(9):
        Bx = B[slave * 3]
        By = B[slave * 3 + 1]
        Bz = B[slave * 3 + 2]
        print('slave {}: {}'.format(slave + 1, (round(Bx, 2), round(By, 2), round(Bz, 2))))
