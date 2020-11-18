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



def readSerial(magOriginDataShare, magFilterDataShare, slavePlot=0):
    port = list(serial.tools.list_ports.comports())[-1][0]
    ser = serial.Serial(port, 9600, timeout=0.5, parity=serial.PARITY_NONE, rtscts=1)
    slaves = 9
    B200 = np.zeros((slaves, 3, 200))
    magOriginData = np.zeros((slaves, 3), dtype=np.int)
    magOffsetData = np.zeros((slaves, 3), dtype=np.int)
    magFilterData = np.zeros((slaves * 3, 1))
    offsetOk = False
    n = 0

    kf = KF(dim_x=slaves * 3, dim_z=slaves * 3)
    kf.P *= 190
    kf.R = np.diag((50, 50, 530) * slaves)
    kf.H = np.eye(slaves * 3)

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
                offsetOk = True
                print('Calibrate ok!')
            else:
                magOriginData -= magOffsetData // 300
                # print('Bx={},By={},Bz={}'.format(magOriginData[4, 0], magOriginData[4, 1], magOriginData[4, 2]))

            magOriginDataShare[:] = np.hstack(magOriginData)[:]
            # 计算200个点的平均值和标准差
            if nn == 0:
                # print('Bx平均值为{:.2f}, Bx方差为{:.2f}, By平均值为{:.2f}, By方差为{:.2f}, Bz平均值为{:.2f}, Bz方差为{:.2f}'.
                #       format(np.mean(B200[slavePlot, 0]), np.var(B200[slavePlot, 0]), np.mean(B200[slavePlot, 1]), np.var(B200[slavePlot, 1]), np.mean(B200[slavePlot, 2]), np.var(B200[slavePlot, 2])))
                B200 = np.zeros((slaves, 3, 200))

            # 使用kalman滤波进行数据平滑
            kf.predict()
            kf.update(magOriginData.reshape(-1, 1))
            magFilterData[:] = kf.x[:]
            magFilterDataShare[:] = np.hstack(magFilterData[:])[:]
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

def plotB(magOriginDataShare, magFilterDataShare, slavePlot=0):
    app = pg.Qt.QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Mag3D Viewer - By Liu Liu")
    win.resize(1500, 500)
    win.setWindowTitle("slave {}: origin VS KF".format(slavePlot))
    pg.setConfigOptions(antialias=True)

    px = win.addPlot(title="Bx")
    px.addLegend()
    px.setLabel('left', 'B', units='mG')
    px.setLabel('bottom', 'points', units='1')
    curve1x = px.plot(pen='r', name='origin')
    curve2x = px.plot(pen='g', name='KF')

    py = win.addPlot(title="By")
    py.addLegend()
    py.setLabel('left', 'B', units='mG')
    py.setLabel('bottom', 'points', units='1')
    curve1y = py.plot(pen='r', name='origin')
    curve2y = py.plot(pen='g', name='KF')

    pz = win.addPlot(title="Bz")
    pz.addLegend()
    pz.setLabel('left', 'B', units='mG')
    pz.setLabel('bottom', 'points', units='1')
    curve1z = pz.plot(pen='r', name='origin')
    curve2z = pz.plot(pen='g', name='KF')


    n, Bx, Bx_KF, By_KF, Bz_KF, By, Bz, i = Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), Queue(), 0

    def update():
        nonlocal i, slavePlot
        i += 1
        n.put(i)
        Bx.put(magOriginDataShare[slavePlot * 3])
        By.put(magOriginDataShare[slavePlot * 3 + 1])
        Bz.put(magOriginDataShare[slavePlot * 3 + 2])
        Bx_KF.put(magFilterDataShare[slavePlot * 3])
        By_KF.put(magFilterDataShare[slavePlot * 3 + 1])
        Bz_KF.put(magFilterDataShare[slavePlot * 3 + 2])

        if i > 100:
            for q in [n, Bx, Bx_KF, By_KF, Bz_KF, By, Bz]:
                q.get()

        for (curve, Bqueue) in [(curve1x, Bx), (curve2x, Bx_KF), (curve1y, By), (curve2y, By_KF), (curve1z, Bz), (curve2z, Bz_KF)]:
            curve.setData(n.queue, Bqueue.queue)

    timer = pg.Qt.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)

    if (sys.flags.interactive != 1) or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()

def complement2origin(x):
    if (x & 0x8000) == 0x8000:
        return -((x - 1) ^ 0xffff)
    else:
        return x

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    slavePlot = 8
    magOriginDataShare = multiprocessing.Array('f', range(27))
    magFilterDataShare = multiprocessing.Array('f', range(27))

    processRead = multiprocessing.Process(target=readSerial, args=(magOriginDataShare, magFilterDataShare, slavePlot))
    processRead.daemon = True
    processRead.start()

    time.sleep(0.5)
    # while True:
    #     plotMag(magOriginDataShare, magFilterDataShare)
    plotB(magOriginDataShare, magFilterDataShare, slavePlot=slavePlot)