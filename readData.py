import numpy as np
from matplotlib import pyplot as plt
import serial
import serial.tools.list_ports
import sys
import time
from queue import Queue
import multiprocessing
from filterpy.kalman import KalmanFilter as KF
import pyqtgraph as pg

SLAVES = 9
MOMENT = 0.3   # 胶囊的磁矩[A*m^2]
DISTANCE = 0.12  # sensor之间的距离[m]
SENSORLOC = np.array(   # sensor的分布
    [[-DISTANCE, DISTANCE, 0], [0, DISTANCE, 0], [DISTANCE, DISTANCE, 0],
     [-DISTANCE, 0, 0], [0, 0, 0], [DISTANCE, 0, 0],
     [-DISTANCE, -DISTANCE, 0], [0, -DISTANCE, 0], [DISTANCE, -DISTANCE, 0]])

def readSerial(magOriginDataShare, magBgDataShare, slavePlot=0):
    port = list(serial.tools.list_ports.comports())[-1][0]
    ser = serial.Serial(port, 9600, timeout=0.5, parity=serial.PARITY_NONE, rtscts=1)
    slaves = 9
    B200 = np.zeros((slaves, 3, 200))
    magOriginData = np.zeros((slaves, 3), dtype=np.int)
    magBgDataShare = np.zeros((slaves, 3), dtype=np.int)
    magOffsetData = np.zeros((slaves, 3), dtype=np.int)
    # magFilterData = np.zeros((slaves * 3, 1))
    offsetOk = False
    n = 0

    # kf = KF(dim_x=slaves * 3, dim_z=slaves * 3)
    # kf.P *= 190
    # kf.Q = np.eye(slaves * 3) * 0.5
    # kf.R = np.diag((15, 15, 15) * slaves)
    # kf.H = np.eye(slaves * 3)

    while True:
        if ser.in_waiting:
            nn = n % 200
            for slave in range(slaves):
                [Bx_L, Bx_H, By_L, By_H, Bz_L, Bz_H, id] = ser.read(7)

                B200[id - 1, 0, nn] = magOriginData[id - 1, 0] = -1.5 * complement2origin((Bx_H << 8) + Bx_L)
                B200[id - 1, 1, nn] = magOriginData[id - 1, 1] = 1.5 * complement2origin((By_H << 8) + By_L)
                B200[id - 1, 2, nn] = magOriginData[id - 1, 2] = 1.5 * complement2origin((Bz_H << 8) + Bz_L)

            # 消除背景磁场
            if (not offsetOk) and n < 300:
                magOffsetData += magOriginData
            elif (not offsetOk) and n == 300:
                magBgDataShare = magOffsetData // 300
                offsetOk = True
                print('Calibrate ok!\nmagBgDataShare-slave5-z={}mGs'.format(magBgDataShare[4, 2]))
            else:
                magOriginData -= magBgDataShare
                # print('Bx={},By={},Bz={}'.format(magOriginData[4, 0], magOriginData[4, 1], magOriginData[4, 2]))

            magOriginDataShare[:] = np.hstack(magOriginData)[:]
            # 计算200个点的平均值和标准差
            if nn == 0:
                # print('Bx平均值为{:.2f}, Bx方差为{:.2f}, By平均值为{:.2f}, By方差为{:.2f}, Bz平均值为{:.2f}, Bz方差为{:.2f}'.
                #       format(np.mean(B200[slavePlot, 0]), np.var(B200[slavePlot, 0]), np.mean(B200[slavePlot, 1]), np.var(B200[slavePlot, 1]), np.mean(B200[slavePlot, 2]), np.var(B200[slavePlot, 2])))
                B200 = np.zeros((slaves, 3, 200))

            # 使用kalman滤波进行数据平滑
            # kf.predict()
            # kf.update(magOriginData.reshape(-1, 1))
            # magFilterData[:] = kf.x[:]
            # magFilterDataShare[:] = np.hstack(magFilterData[:])[:]
            n += 1

def plotMag(magOriginData, magFilterData):
    t, Bx, Bx_KF, By_KF, Bz_KF, By, Bz, i = Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), 0
    plt.ion()

    # fig, ax = plt.subplots(nrows=1)
    fig, axs = plt.subplots(nrows=3)
    while True:
        slavePlot = 4
        i += 1
        t.put(i)
        Bx.put(magOriginData[slavePlot * 3])
        By.put(magOriginData[slavePlot * 3 + 1])
        Bz.put(magOriginData[slavePlot * 3 + 2])
        Bx_KF.put(magFilterData[slavePlot * 3])
        By_KF.put(magFilterData[slavePlot * 3 + 1])
        Bz_KF.put(magFilterData[slavePlot * 3 + 2])

        # 数据超过50个以后刷新
        if i > 50:
            t.get()
            Bx.get()
            By.get()
            Bz.get()
            Bx_KF.get()
            By_KF.get()
            Bz_KF.get()

        # ax.cla()
        # ax.plot(t, Bx, 'red', label='origin')
        # ax.plot(t, Bx_KF, 'green', label='KF')
        # ax.set_title(label='B',loc='left')
        # ax.legend(loc=2)

        for ax in axs:
            ax.cla()
        axs[0].plot(t.queue, Bx.queue, 'red', label='origin')
        axs[0].plot(t.queue, Bx_KF.queue, 'green', label='KF')
        axs[0].set_title(label='Bx',loc='left')
        axs[1].plot(t.queue, By.queue, 'blue', label='origin')
        axs[1].plot(t.queue, By_KF.queue, 'green', label='KF')
        axs[1].set_title(label='By', loc='left')
        axs[2].plot(t.queue, Bz.queue, 'black', label='origin')
        axs[2].plot(t.queue, Bz_KF.queue, 'green', label='KF')
        axs[2].set_title(label='Bz', loc='left')
        for ax in axs:
            ax.legend(loc=2)
        plt.pause(0.001)

def plotB(magOriginDataShare , slavePlot=(1, 5, 9), state=None):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer")
    win.resize(1500, 900)
    win.setWindowTitle("slave {}: origin VS KF".format(slavePlot))
    pg.setConfigOptions(antialias=True)

    n = Queue()
    curves = []  
    datas = []   # [s1_Bx_Origin, s1_Bx_Predict, s1_By_Origin, s1_By_Predict, ... ]
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
            datas.append(Queue())    # origin
            datas.append(Queue())    # Predict
        win.nextRow()
    i = 0

    def update():
        nonlocal i
        magPredictData = h(state) if state else np.zeros(27)
        i += 1
        n.put(i)
        for slaveIndex, slave in enumerate(slavePlot):
            for Bindex in range(3):
                datas[slaveIndex * 6 + Bindex * 2].put(magOriginDataShare[(slave-1) * 3 + Bindex])
                datas[slaveIndex * 6 + Bindex * 2 + 1].put(magPredictData[(slave-1) * 3 + Bindex])

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
    x, y, z = state[0:6:2]
    q0, q1, q2, q3 = state[6:10]
    mNorm = np.array([q2m(q0, q1, q2, q3)])
    rotNorm = np.array([q2m(q0, q1, q2, q3)] * 9)

    pos = np.array([[x, y, z]] * 9) - SENSORLOC
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    posNorm = pos / r

    B = MOMENT * np.multiply(r ** (-3), np.subtract(3 * np.multiply(np.inner(posNorm, mNorm), posNorm), rotNorm)) # 每个sensor的B值[mGs]
    data = B.reshape(-1)
    return data

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    slavePlot = (1, 5, 9)
    magOriginDataShare = multiprocessing.Array('f', range(27))
    magFilterDataShare = multiprocessing.Array('f', range(27))

    processRead = multiprocessing.Process(target=readSerial, args=(magOriginDataShare, slavePlot))
    processRead.daemon = True
    processRead.start()

    time.sleep(0.5)
    # while True:
    #     plotMag(magOriginDataShare, magFilterDataShare)
    plotB(magOriginDataShare, slavePlot=slavePlot)