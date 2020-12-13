import json
import math
import numpy as np
import serial
import serial.tools.list_ports
import sys
import time
from queue import Queue
import multiprocessing
from filterpy.kalman import FixedLagSmoother as FLS
import pyqtgraph as pg

SLAVES = 9
MOMENT = 0.3  # 胶囊的磁矩[A*m^2]
DISTANCE = 0.12  # sensor之间的距离[m]
SENSORLOC = np.array(  # sensor的分布
    [[-DISTANCE, DISTANCE, 0], [0, DISTANCE, 0], [DISTANCE, DISTANCE, 0],
     [-DISTANCE, 0, 0], [0, 0, 0], [DISTANCE, 0, 0],
     [-DISTANCE, -DISTANCE, 0], [0, -DISTANCE, 0], [DISTANCE, -DISTANCE, 0]])


def readSerial(magOriginData, magSmoothData, slavePlot=0):
    port = list(serial.tools.list_ports.comports())[-1][0]
    ser = serial.Serial(port, 9600, timeout=0.5, parity=serial.PARITY_NONE, rtscts=1)
    slaves = 9
    B200 = np.zeros((slaves, 3, 200))
    OriginData = np.zeros((slaves, 3), dtype=np.float)
    magBgData = np.zeros((slaves, 3), dtype=np.float)
    magOffsetData = np.zeros((slaves, 3), dtype=np.float)
    offsetOk = True
    n = 0

    fls = FLS(dim_x=SLAVES * 3, dim_z=SLAVES * 3, N=10)
    fls.P *= 200
    fls.R *= 50
    fls.Q *= 0.5

    if offsetOk:
        f = open('bg.json', 'r')
        bg = json.load(f)
        for row in range(SLAVES):
            for col in range(3):
                magBgData[row, col] = bg.get('B{}{}'.format(row, col), 0)
        f.close()
        print('get background B OK!')

    while True:
        if ser.in_waiting:
            nn = n % 200
            for slave in range(slaves):
                [Bx_L, Bx_H, By_L, By_H, Bz_L, Bz_H, id] = ser.read(7)

                B200[id - 1, 0, nn] = OriginData[id - 1, 0] = -1.5 * complement2origin((Bx_H << 8) + Bx_L)
                B200[id - 1, 1, nn] = OriginData[id - 1, 1] = 1.5 * complement2origin((By_H << 8) + By_L)
                B200[id - 1, 2, nn] = OriginData[id - 1, 2] = 1.5 * complement2origin((Bz_H << 8) + Bz_L)

            # 消除背景磁场
            if (not offsetOk) and n < 300:
                magOffsetData += OriginData
            elif (not offsetOk) and n == 300:
                magBgData = magOffsetData // 300
                offsetOk = True
                print('Calibrate ok!')

                # 保存背景磁场到本地json文件
                bg = {}
                for row in range(SLAVES):
                    for col in range(3):
                        bg['B{}{}'.format(row, col)] = magBgData[row, col]
                f = open('bg.json', 'w')
                json.dump(bg, f, indent=4)
                f.close()
            else:
                OriginData -= magBgData

            magOriginData[:] = np.hstack(OriginData)[:]
            # 计算200个点的平均值和标准差
            if nn == 0:
                # print('Bx平均值为{:.2f}, Bx方差为{:.2f}, By平均值为{:.2f}, By方差为{:.2f}, Bz平均值为{:.2f}, Bz方差为{:.2f}'.
                #       format(np.mean(B200[slavePlot, 0]), np.var(B200[slavePlot, 0]), np.mean(B200[slavePlot, 1]), np.var(B200[slavePlot, 1]), np.mean(B200[slavePlot, 2]), np.var(B200[slavePlot, 2])))
                B200 = np.zeros((slaves, 3, 200))

            # 使用FixedLagSmoother对原始数据进行平滑
            fls.smooth(magOriginData[:])
            tmp = np.array(fls.xSmooth[0])
            magSmoothData[:] = np.array(fls.xSmooth[-1])[0, :]
            n += 1


def plotMag(magOriginData, magSmoothData, slavePlot=(1, 5, 9)):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer")
    win.resize(1500, 900)
    win.setWindowTitle("slave {}: origin VS KF".format(slavePlot))
    pg.setConfigOptions(antialias=True)

    n = Queue()
    curves = []
    datas = []  # [s1_Bx_Origin, s1_Bx_Smooth, s1_By_Origin, s1_By_Smooth, ... ]
    for i in slavePlot:
        for Bi in ['Bx', 'By', 'Bz']:
            p = win.addPlot(title='slave {}--'.format(i) + Bi)
            p.addLegend()
            p.setLabel('left', 'B', units='mG')
            p.setLabel('bottom', 'points', units='1')
            cOrigin = p.plot(pen='r', name='Origin')
            cPredict = p.plot(pen='g', name='Smooth')
            curves.append(cOrigin)
            curves.append(cPredict)
            datas.append(Queue())  # origin
            datas.append(Queue())  # Predict
        win.nextRow()
    i = 0

    def update():
        nonlocal i
        i += 1
        n.put(i)
        for slaveIndex, slave in enumerate(slavePlot):
            for Bindex in range(3):
                datas[slaveIndex * 6 + Bindex * 2].put(magOriginData[(slave - 1) * 3 + Bindex])
                datas[slaveIndex * 6 + Bindex * 2 + 1].put(magSmoothData[(slave - 1) * 3 + Bindex])

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


def plotB(magOriginData, slavePlot=(1, 5, 9), state=None):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer")
    win.resize(1500, 900)
    win.setWindowTitle("slave {}: origin VS KF".format(slavePlot))
    pg.setConfigOptions(antialias=True)

    n = Queue()
    curves = []
    datas = []  # [s1_Bx_Origin, s1_Bx_Predict, s1_By_Origin, s1_By_Predict, ... ]
    for i in slavePlot:
        for Bi in ['Bx', 'By', 'Bz']:
            p = win.addPlot(title='slave {}--'.format(i) + Bi)
            p.addLegend()
            p.setLabel('left', 'B', units='mG')
            p.setLabel('bottom', 'points', units='1')
            cOrigin = p.plot(pen='r', name='Origin')
            cPredict = p.plot(pen='g', name='Predict')
            curves.append(cOrigin)
            curves.append(cPredict)
            datas.append(Queue())  # origin
            datas.append(Queue())  # Predict
        win.nextRow()
    i = 0

    def update():
        nonlocal i
        magPredictData = h(state) if state else np.zeros(27)
        i += 1
        n.put(i)
        for slaveIndex, slave in enumerate(slavePlot):
            for Bindex in range(3):
                datas[slaveIndex * 6 + Bindex * 2].put(magOriginData[(slave - 1) * 3 + Bindex])
                datas[slaveIndex * 6 + Bindex * 2 + 1].put(magPredictData[(slave - 1) * 3 + Bindex])

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


def complement2origin(x):
    if (x & 0x8000) == 0x8000:
        return -((x - 1) ^ 0xffff)
    else:
        return x


def q2m(q0, q1, q2, q3):
    qq2 = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    mx = 2 * (-q0 * q2 + q1 * q3) / qq2
    my = 2 * (q0 * q1 + q2 * q3) / qq2
    mz = (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) / qq2
    return [round(mx, 2), round(my, 2), round(mz, 2)]


def h(state):
    B = np.zeros((9, 3))
    x, y, z, q0, q1, q2, q3 = state[0: 7]
    mNorm = np.array([q2m(q0, q1, q2, q3)])
    rotNorm = np.array([q2m(q0, q1, q2, q3)] * 9)

    pos = np.array([[x, y, z]] * 9) - SENSORLOC
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    posNorm = pos / r

    B = MOMENT * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm),
                                                    rotNorm))  # 每个sensor的B值[mGs]
    data = B.reshape(-1)
    return data


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    slavePlot = (1, 5, 9)
    magOriginData = multiprocessing.Array('f', range(27))
    magSmoothData = multiprocessing.Array('f', range(27))

    processRead = multiprocessing.Process(target=readSerial, args=(magOriginData, magSmoothData, slavePlot))
    processRead.daemon = True
    processRead.start()

    time.sleep(0.5)
    while True:
        plotMag(magOriginData, magSmoothData)
    # plotB(magOriginData, slavePlot=slavePlot)
