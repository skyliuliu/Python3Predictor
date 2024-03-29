import json
import numpy as np
import serial
import serial.tools.list_ports
import sys
import time
from queue import Queue
import multiprocessing
from filterpy.kalman import FixedLagSmoother as FLS
import pyqtgraph as pg
from scipy import stats

SLAVES = 9  # sensor个数
MOMENT = 0.3  # 胶囊的磁矩[A*m^2]
DISTANCE = 0.12  # sensor之间的距离[m]
SENSORLOC = np.array(  # sensor的分布
    [[-DISTANCE, DISTANCE, 0], [0, DISTANCE, 0], [DISTANCE, DISTANCE, 0],
     [-DISTANCE, 0, 0], [0, 0, 0], [DISTANCE, 0, 0],
     [-DISTANCE, -DISTANCE, 0], [0, -DISTANCE, 0], [DISTANCE, -DISTANCE, 0]])


def readSerial(B0, Bs, Bg, slavePlot=0):
    port = list(serial.tools.list_ports.comports())[-1][0]
    ser = serial.Serial(port, 9600, timeout=0.5, parity=serial.PARITY_NONE, rtscts=1)
    navg = 10  # 均值平滑的数据个数
    Bnavg = np.zeros((SLAVES, 3))  # 均值平滑储存的数据
    nmax = 200  # 为计算标准差采集数据个数
    Bnmax = np.zeros((SLAVES, 3, nmax))  # 计算标准差储存的数据

    OriginData = np.zeros((SLAVES, 3), dtype=np.float)
    magOffsetData = np.zeros((SLAVES, 3), dtype=np.float)
    offsetOk = True
    n = 0
    # 固定区间平滑器
    fls = FLS(dim_x=SLAVES * 3, dim_z=SLAVES * 3, N=20)
    fls.P *= 200
    fls.R *= 50
    fls.Q *= 0.5
    # 使用闭包来评估数据是否满足正态分布
    ftestNormal = testNormal
    # 读取本地保存的背景磁场
    if offsetOk:
        f = open('bg.json', 'r')
        bg = json.load(f)
        for row in range(SLAVES):
            for col in range(3):
                Bg[row*3 + col] = bg.get('B{}{}'.format(row, col), 0)
        f.close()
        print('get background B OK!')
    # 持续读取sensor数据
    while True:
        if ser.in_waiting:
            nn = n % nmax
            for slave in range(SLAVES):
                [Bx_L, Bx_H, By_L, By_H, Bz_L, Bz_H, id] = ser.read(7)
                OriginData[id - 1, 0] = -1.5 * complement2origin((Bx_H << 8) + Bx_L)
                OriginData[id - 1, 1] = 1.5 * complement2origin((By_H << 8) + By_L)
                OriginData[id - 1, 2] = 1.5 * complement2origin((Bz_H << 8) + Bz_L)

            # 扣除背景磁场
            if (not offsetOk) and n < 300:
                magOffsetData += OriginData
            elif (not offsetOk) and n == 300:
                Bg[:] = magOffsetData.reshape(-1) // 300
                offsetOk = True
                print('Calibrate ok!')

                # 保存背景磁场到本地json文件
                bg = {}
                for row in range(SLAVES):
                    for col in range(3):
                        bg['B{}{}'.format(row, col)] = Bg[row*3 + col]
                f = open('bg.json', 'w')
                json.dump(bg, f, indent=4)
                f.close()
            else:
                OriginData -= np.array(Bg).reshape(9, 3)

            B0[:] = np.hstack(OriginData)[:]

            # 使用FixedLagSmoother对原始数据进行平滑
            fls.smooth(B0[:])
            tmp = np.array(fls.xSmooth[0])
            Bs[:] = np.array(fls.xSmooth[-1])[0, :]

            # 取navg个点的平均值进行平滑
            # Bnavg += OriginData
            # if n % navg == 0:
            #     Bs[:] = np.hstack(Bnavg // navg)[:]
            #     Bnavg = np.zeros((SLAVES, 3))
            #     Bnmax[slavePlot, 2, (n // navg) % (nmax // navg)] = Bs[14]

            # 计算nmax个点的平均值和标准差
            # Bnmax[slavePlot, 1, nn] = Bs[slavePlot * 3 + 1]
            # if nn == 0 and n > 0:
            #     ftestNormal(Bnmax[slavePlot, 1])
            #     Bnmax = np.zeros((SLAVES, 3, nmax))

            n += 1


def plotMag(B0, Bs, slavePlot=(1, 5, 9)):
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
            datas.append(Queue())  # smooth
        win.nextRow()
    i = 0

    def update():
        nonlocal i
        i += 1
        n.put(i)
        for slaveIndex, slave in enumerate(slavePlot):
            for Bindex in range(3):
                datas[slaveIndex * 6 + Bindex * 2].put(B0[(slave - 1) * 3 + Bindex])
                datas[slaveIndex * 6 + Bindex * 2 + 1].put(Bs[(slave - 1) * 3 + Bindex])

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

def plotMagDelta(B0, Bs, slavePlot=(4, 5, 6)):
    # 不同sensor之间的B值之差
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
            p = win.addPlot(title='slave {}-{} {}'.format(i, i + 1, Bi))
            p.addLegend()
            p.setLabel('left', 'B', units='mG')
            p.setLabel('bottom', 'points', units='1')
            cOrigin = p.plot(pen='r', name='Origin')
            cPredict = p.plot(pen='g', name='Smooth')
            curves.append(cOrigin)
            curves.append(cPredict)
            datas.append(Queue())  # origin
            datas.append(Queue())  # smooth
        win.nextRow()
    i = 0

    def update():
        nonlocal i
        i += 1
        n.put(i)
        for slaveIndex, slave in enumerate(slavePlot):
            for Bindex in range(3):
                B0_delta = B0[(slave - 1) * 3 + Bindex] - B0[slave * 3 + Bindex]
                Bs_delta = Bs[(slave - 1) * 3 + Bindex] - Bs[slave * 3 + Bindex]
                datas[slaveIndex * 6 + Bindex * 2].put(B0_delta)
                datas[slaveIndex * 6 + Bindex * 2 + 1].put(Bs_delta)

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


def plotB(B0, slavePlot=(1, 5, 9), state=None):
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
                datas[slaveIndex * 6 + Bindex * 2].put(B0[(slave - 1) * 3 + Bindex])
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


def testNormal(data):
    mean, std = np.mean(data), np.std(data)
    r = stats.kstest(data, 'norm', args=(mean, std))
    print('r={}, mean={}, var={}'.format(r, mean, std * std))
    # plt.hist(data, bins=100, histtype='bar', rwidth=0.8)
    # plt.show()


def complement2origin(x):
    if (x & 0x8000) == 0x8000:
        return -((x - 1) ^ 0xffff)
    else:
        return x


def q2m(q0, q1, q2, q3):
    qq2 = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    mx = 2 * (q0 * q2 + q1 * q3) / qq2
    my = 2 * (-q0 * q1 + q2 * q3) / qq2
    mz = (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) / qq2
    return [round(mx, 3), round(my, 3), round(mz, 3)]


def h(state):
    B = np.zeros((9, 3))
    x, y, z, q0, q1, q2, q3 = state[0: 7]
    mNorm = np.array([q2m(q0, q1, q2, q3)])
    rotNorm = np.array([q2m(q0, q1, q2, q3)] * 9)

    pos = np.array([[x, y, z]] * 9) - SENSORLOC
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    posNorm = pos / r

    B[:] = MOMENT * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm),
                                                    rotNorm))  # 每个sensor的B值[mGs]
    data = B.reshape(-1)
    return data


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    slavePlot = (1, 5, 9)
    B0 = multiprocessing.Array('f', range(27))
    Bs = multiprocessing.Array('f', range(27))
    Bg = multiprocessing.Array('f', range(27))

    processRead = multiprocessing.Process(target=readSerial, args=(B0, Bs, Bg))
    processRead.daemon = True
    processRead.start()

    time.sleep(0.5)
    plotMag(B0, Bs)
