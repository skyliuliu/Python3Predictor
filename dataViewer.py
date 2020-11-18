from PyQt5.QtWidgets import QWidget
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from PyQt5.QtCore import QRect
import sys
import pyqtgraph.opengl as gl
from math import sqrt, cos, sin, pi


class Mag3DViewer(QWidget):
    "3D Viewer Widget for the plotter"
    def __init__(self, nMAG):
        QWidget.__init__(self)
        self.MagNum = nMAG
        self.resize(800, 700)
        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap(_fromUtf8("QtIcon/png/Safari Black.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        # self.setWindowIcon(icon)
        self.setWindowTitle('Mag3D Viewer - By Liu Liu')
        self.Area = DockArea()

        self.h = QtGui.QHBoxLayout()
        self.setLayout(self.h)
        self.h.addWidget(self.Area)
        self.initGL()
        self.initStatus()
        self.TwoTrans = np.zeros((2, 3))
        self.show()
        self._isPause = False
        self.text_Postext, self.text_Rottext, self.text_Mtext = "\n", "\n", "\n"
        self.MagPosXYZ = np.zeros((3, self.MagNum))
        self.MagMoment = np.zeros(self.MagNum)
        self.MomentXYZ = np.zeros((3, self.MagNum))
        self.MagPolarAngle = np.zeros((2, self.MagNum))
        self.Timecost = 0.0

    def createYaxis(self, father):
        # y-axis
        y_axis = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1, 0.1], length=44.)
        y_axis = gl.GLMeshItem(meshdata=y_axis, smooth=False, color=(0, 0, 1, 0.7), shader='balloon', glOptions='additive')
        father.addItem(y_axis)
        y_axis.rotate(-90., 0, 1, 0)
        y_axis.translate(22, 0, 0)
        y_axis_arrow = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2.)
        y_axis_arrow = gl.GLMeshItem(meshdata=y_axis_arrow, smooth=False, color=(0, 0, 1, 0.7), shader='balloon', glOptions='additive')
        father.addItem(y_axis_arrow)
        y_axis_arrow.rotate(-90., 0, 1, 0)
        y_axis_arrow.translate(-22, 0, 0)
        # x word
        y_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=2)
        y_axis_word = gl.GLMeshItem(meshdata=y_axis_word, smooth=False, color=(0, 0, 1, 0.7), shader='balloon', glOptions='additive')
        father.addItem(y_axis_word)
        y_axis_word.rotate(-45., 1, 0, 0)
        y_axis_word.translate(-25, 0, 2)

        y_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=2)
        y_axis_word = gl.GLMeshItem(meshdata=y_axis_word, smooth=False, color=(0, 0, 1, 0.7), shader='balloon', glOptions='additive')
        father.addItem(y_axis_word)
        y_axis_word.rotate(45., 1, 0, 0)
        y_axis_word.translate(-25, 0, 2)

        y_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=2)
        y_axis_word = gl.GLMeshItem(meshdata=y_axis_word, smooth=False, color=(0, 0, 1, 0.7), shader='balloon', glOptions='additive')
        father.addItem(y_axis_word)
        y_axis_word.translate(-25, 0, 0)

    def createXaxis(self, father):
        # x-axis
        x_axis = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1, 0.1], length=44.)
        x_axis = gl.GLMeshItem(meshdata=x_axis, smooth=False, color=(0, 1, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(x_axis)
        x_axis.rotate(-90., 1, 0, 0)
        x_axis.translate(0,-22, 0)
        x_axis_arrow = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2.)
        x_axis_arrow = gl.GLMeshItem(meshdata=x_axis_arrow, smooth=False, color=(0, 1, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(x_axis_arrow)
        x_axis_arrow.rotate(-90., 1, 0, 0)
        x_axis_arrow.translate(0, 22, 0)

        x_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=3)
        x_axis_word = gl.GLMeshItem(meshdata=x_axis_word, smooth=False, color=(0, 1, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(x_axis_word)
        x_axis_word.rotate(-30, 1, 0, 0)
        x_axis_word.translate(0, 25, 0)

        x_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=3)
        x_axis_word = gl.GLMeshItem(meshdata=x_axis_word, smooth=False, color=(0, 1, 0, 0.7),shader='balloon', glOptions='additive')
        father.addItem(x_axis_word)
        x_axis_word.rotate(30, 1, 0, 0)
        x_axis_word.translate(0, 26.5, 0)

    def createZaxis(self, father):
        z_axis = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1, 0.1], length=18.)
        z_axis = gl.GLMeshItem(meshdata=z_axis, smooth=True, color=(1, 0, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(z_axis)
        z_axis_arrow = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2.)
        z_axis_arrow = gl.GLMeshItem(meshdata=z_axis_arrow, smooth=True, color=(1, 0, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(z_axis_arrow)
        z_axis_arrow.translate(0, 0, 18)

        z_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=2)
        z_axis_word = gl.GLMeshItem(meshdata=z_axis_word, smooth=True, color=(1, 0, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(z_axis_word)
        z_axis_word.rotate(-90., 1, 0, 0)
        z_axis_word.translate(0, -1, 21)

        z_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=2)
        z_axis_word = gl.GLMeshItem(meshdata=z_axis_word, smooth=True, color=(1, 0, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(z_axis_word)
        z_axis_word.rotate(-90., 1, 0, 0)
        z_axis_word.translate(0, -1, 23)

        z_axis_word = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.2, 0.2], length=2*sqrt(2))
        z_axis_word = gl.GLMeshItem(meshdata=z_axis_word, smooth=True, color=(1, 0, 0, 0.7), shader='balloon', glOptions='additive')
        father.addItem(z_axis_word)
        z_axis_word.rotate(-45., 1, 0, 0)
        z_axis_word.translate(0, -1, 21)

    def initGL(self):
        self.GLDock = Dock("3D Magnetic dipoles ",size=(500, 400))
        w = gl.GLViewWidget()
        self.GLDock.addWidget(w)
        self.GLDock.hideTitleBar()
        self.Area.addDock(self.GLDock, 'left')
        w.show()
        w.setCameraPosition(distance=60)
        g = gl.GLGridItem()
        g.scale(2, 2, 1)
        w.addItem(g)

        self.createXaxis(w)
        self.createYaxis(w)
        self.createZaxis(w)

        self.sphereData = []
        self.sphereMesh = []
        self.ArrowData = []
        self.ArrowMesh= []
        for i in range(self.MagNum):
            sphereData = gl.MeshData.sphere(rows=10, cols=20, radius = 0.8)
            sphereMesh = gl.GLMeshItem(meshdata=sphereData, smooth=True, shader='shaded', glOptions='opaque')
            w.addItem(sphereMesh)
            self.sphereMesh.append(sphereMesh)
            ArrowData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.7, 0.], length=1.5)
            ArrowMesh = gl.GLMeshItem(meshdata=ArrowData, smooth=True, color=(1, 0, 0, 0.7), shader='balloon', glOptions='opaque')
            ArrowMesh.translate(0, 0, 1)
            ArrowMesh.rotate(0., 0, 1, 1)
            w.addItem(ArrowMesh)
            self.ArrowMesh.append(ArrowMesh)

    def window_showData(self):
        self.text_Postext, self.text_Rottext, self.text_Mtext = "", "", ""
        for i in range(self.MagNum):
            self.text_Postext += "   x=%.1f y=%.1f y=%.1f \n" % (self.MagPosXYZ[0, i], self.MagPosXYZ[1, i], self.MagPosXYZ[2, i])
            self.text_Rottext += "   theta=%.2f, phi=%.2f  \n" % (self.MagPolarAngle[0, i], self.MagPolarAngle[1, i])
            self.text_Mtext += "  %.2f\n" % self.MagMoment[i]
        self.text_Timetext = "   %.3f (s)" % self.Timecost
        self.PosText.setText(self.text_Postext)
        self.RotText.setText(self.text_Rottext)
        self.MomentText.setText(self.text_Mtext)
        self.TimeText.setText(self.text_Timetext)

    def initStatus(self):
        self.PosText = QtGui.QLabel()
        PosDock = Dock("Position (cm)",size=(1, 1))
        PosDock.addWidget(self.PosText)
        self.Area.addDock(PosDock, 'right')

        self.RotText = QtGui.QLabel()
        RotDock = Dock("Rotation (degrees)",size=(150, 1))
        RotDock.addWidget(self.RotText)
        self.Area.addDock(RotDock, 'bottom', PosDock)

        self.MomentText = QtGui.QLabel()
        MomentDock = Dock("Moment (A*m^2)", size=(150, 1))
        MomentDock.addWidget(self.MomentText)
        self.Area.addDock(MomentDock, 'bottom', RotDock)

        self.TimeText = QtGui.QLabel()
        TimeDock = Dock("Time Cost (s)", size=(150, 1))
        TimeDock.addWidget(self.TimeText)
        self.Area.addDock(TimeDock, 'bottom',MomentDock)

        self.PauseBtn = QtGui.QPushButton('Pause')
        self.PauseBtn.setEnabled(True)
        self.PauseBtn.clicked.connect(self.Pause)

        BtnDock = Dock("Time Cost", size=(150, 1))
        BtnDock.hideTitleBar()
        l = pg.LayoutWidget()
        BtnDock.addWidget(l)
        l.addWidget(self.PauseBtn,row=0, col=0)
        self.Area.addDock(BtnDock, 'bottom',TimeDock)

    def Pause(self):
        self._isPause = not self._isPause
        if self._isPause:
            self.PauseBtn.setText('Restart')
        else:
            self.PauseBtn.setText('Pause')

    def onRender(self, state):
        if not self._isPause:
            for i in range(self.MagNum):
                a, b, c, q0, q1, q2, q3, m = state[8 * i:8 * i + 8]
                axis, angle = qtoAxisAngle(q0, q1, q2, q3)
                self.sphereMesh[i].resetTransform()
                self.sphereMesh[i].rotate(angle, -axis[1], axis[0], axis[2])
                self.sphereMesh[i].translate(-b, a, c)

                self.ArrowMesh[i].resetTransform()
                self.ArrowMesh[i].rotate(angle, -axis[1], axis[0], axis[2])
                self.ArrowMesh[i].translate(-b, a, c)
                self.MagPosXYZ[0:3, i] = a, b, c
                self.MagMoment[i] = m
                self.MagPolarAngle[0:2, i] = getMoment_PolarAngle(q0, q1, q2, q3)
            self.Timecost = state[-1]
            self.window_showData()

def qtoAxisAngle(q0, q1, q2, q3):
    x = (2 * q1 * q3 - 2 * q0 * q2) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    y = (2 * q2 * q3 + 2 * q0 * q1) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    z = (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    angle = np.degrees(np.arccos(z))  # 先得到theta角
    axis = np.array([-y/sqrt(x**2+y**2), x/sqrt(x**2+y**2), 0])
    return axis, angle

def getMoment_PolarAngle(q0, q1, q2, q3, realize=False):
    """Get polar coordinate angle theta, phi"""
    x = (2 * q1 * q3 - 2 * q0 * q2) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    y = (2 * q2 * q3 + 2 * q0 * q1) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    z = (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
    theta = np.degrees(np.arccos(z))  # 先得到theta角
    if theta == 0:
        phi = 0
    else:
        if y < 0:  # 如果在xy三四象限
            phi = -np.degrees(np.arccos(x/pow((x**2+y**2), 0.5)))
        else:  # 如果在xy一二象限
            phi = np.degrees(np.arccos(x/pow((x**2+y**2), 0.5)))
    return np.array([theta, phi])

def magViewer(mp):
    app = QtGui.QApplication([])
    magViewer = Mag3DViewer(1)
    magViewer.setGeometry(QRect(800, 200, 900, 600))
    # magViewer.show()
    def updateData():
        state_pos = np.concatenate((mp.ukf.x[0: 3] * 100, mp.ukf.x[6: 10]))
        state = np.append(state_pos, mp.moment)
        magViewer.onRender(state)

    t = QtCore.QTimer()
    t.timeout.connect(updateData)
    t.start(50)
    # sys.exit(app.exec())
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()