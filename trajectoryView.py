import math
import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
import OpenGL.GL as ogl
import numpy as np


class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, X, Y, Z, text):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(int(self.X), int(self.Y), int(self.Z), self.text)


class Custom3DAxis(gl.GLAxisItem):
    """Class defined to extend 'gl.GLAxisItem'."""

    def __init__(self, parent, color=(1, 2, 3, 4)):
        gl.GLAxisItem.__init__(self)
        self.parent = parent
        self.c = color
        self.ticks = [-20, -10, 0, 10, 20]
        self.setSize(x=40, y=40, z=40)
        self.add_labels()
        self.add_tick_values(xticks=self.ticks, yticks=self.ticks, zticks=[0, 10, 20, 30, 40])
        self.addArrow()

    def add_labels(self):
        """Adds axes labels."""
        x, y, z = self.size()
        x *= 0.5
        y *= 0.5
        # X label
        self.xLabel = CustomTextItem(X=x + 0.5, Y=-y / 10, Z=-z / 10, text="X(cm)")
        self.xLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.xLabel)
        # Y label
        self.yLabel = CustomTextItem(X=-x / 10, Y=y + 0.5, Z=-z / 10, text="Y(cm)")
        self.yLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.yLabel)
        # Z label
        self.zLabel = CustomTextItem(X=-x / 10, Y=-y / 10, Z=z + 1, text="Z(cm)")
        self.zLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.zLabel)

    def add_tick_values(self, xticks=None, yticks=None, zticks=None):
        """Adds ticks values."""
        x, y, z = self.size()
        xtpos = np.linspace(-0.5 * x, 0.5 * x, len(xticks))
        ytpos = np.linspace(-0.5 * y, 0.5 * y, len(yticks))
        ztpos = np.linspace(0, z, len(zticks))
        # X label
        for i, xt in enumerate(xticks):
            val = CustomTextItem(X=xtpos[i], Y=2, Z=0, text=str(xt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)
        # Y label
        for i, yt in enumerate(yticks):
            val = CustomTextItem(X=2, Y=ytpos[i], Z=0, text=str(yt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)
        # Z label
        for i, zt in enumerate(zticks):
            val = CustomTextItem(X=0, Y=2, Z=ztpos[i], text=str(zt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)

    def addArrow(self):
        # add X axis arrow
        arrowXData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowX = gl.GLMeshItem(meshdata=arrowXData, color=(0, 0, 1, 0.6), shader='balloon', glOptions='opaque')
        arrowX.rotate(90, 0, 1, 0)
        arrowX.translate(20, 0, 0)
        self.parent.addItem(arrowX)
        # add Y axis arrow
        arrowYData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowY = gl.GLMeshItem(meshdata=arrowXData, color=(1, 0, 1, 0.6), shader='balloon', glOptions='opaque')
        arrowY.rotate(270, 1, 0, 0)
        arrowY.translate(0, 20, 0)
        self.parent.addItem(arrowY)
        # add Z axis arrow
        arrowZData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
        arrowZ = gl.GLMeshItem(meshdata=arrowXData, color=(0, 1, 0, 0.6), shader='balloon', glOptions='opaque')
        arrowZ.translate(0, 0, 40)
        self.parent.addItem(arrowZ)

    def paint(self):
        self.setupGLState()
        if self.antialias:
            ogl.glEnable(ogl.GL_LINE_SMOOTH)
            ogl.glHint(ogl.GL_LINE_SMOOTH_HINT, ogl.GL_NICEST)
        ogl.glBegin(ogl.GL_LINES)

        x, y, z = self.size()
        # Draw Z
        ogl.glColor4f(0, 1, 0, 10.6)  # z is green
        ogl.glVertex3f(0, 0, 0)
        ogl.glVertex3f(0, 0, z)
        # Draw Y
        ogl.glColor4f(1, 0, 1, 10.6)  # y is grape
        ogl.glVertex3f(0, -0.5 * y, 0)
        ogl.glVertex3f(0, 0.5 * y, 0)
        # Draw X
        ogl.glColor4f(0, 0, 1, 10.6)  # x is blue
        ogl.glVertex3f(-0.5 * x, 0, 0)
        ogl.glVertex3f(0.5 * x, 0, 0)
        ogl.glEnd()


def q2ua(q0, q1, q2, q3):
    qq = np.linalg.norm([q0, q1, q2, q3])
    q0 /= qq
    q1 /= qq
    q2 /= qq
    q3 /= qq
    angle = 2 * math.acos(q0)
    u = np.array([q1, q2, q3]) / math.sin(0.5 * angle)
    return u, angle * 57.3


def track3D(state):
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.setWindowTitle('3d trajectory')
    w.resize(600, 500)
    # instance of Custom3DAxis
    axis = Custom3DAxis(w, color=(0.6, 0.6, 0.2, .6))
    w.addItem(axis)
    w.opts['distance'] = 75
    w.opts['center'] = Vector(0, 0, 15)
    # add xy grid
    gx = gl.GLGridItem()
    gx.setSize(x=40, y=40, z=10)
    gx.setSpacing(x=5, y=5)
    w.addItem(gx)
    # trajectory line
    pos0 = np.array([[0, 0, 0]])
    pos, q = np.array(state[:3]), state[3:7]
    uAxis, angle = q2ua(*q)
    track0 = np.concatenate((pos0, pos.reshape(1, 3)))
    plt = gl.GLLinePlotItem(pos=track0, width=2, color=(1, 0, 0, .6))
    w.addItem(plt)
    # orientation arrow
    sphereData = gl.MeshData.sphere(rows=10, cols=20, radius=0.8)
    sphereMesh = gl.GLMeshItem(meshdata=sphereData, smooth=True, shader='shaded', glOptions='opaque')
    w.addItem(sphereMesh)
    ArrowData = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.5, 0.], length=2)
    ArrowMesh = gl.GLMeshItem(meshdata=ArrowData, smooth=True, color=(1, 0, 0, 0.6), shader='balloon',
                              glOptions='opaque')
    ArrowMesh.rotate(90, 0, 1, 0)
    w.addItem(ArrowMesh)
    w.show()

    i = 1
    pts = pos.reshape(1, 3)

    def update():
        '''update position and orientation'''
        nonlocal i, pts, state
        pos, q = np.array(state[:3]) * 100, state[3:7]
        uAxis, angle = q2ua(*q)
        pt = (pos).reshape(1, 3)
        if pts.size < 150:
            pts = np.concatenate((pts, pt))
        else:
            pts = np.concatenate((pts[-50:, :], pt))
        plt.setData(pos=pts)
        ArrowMesh.resetTransform()
        sphereMesh.resetTransform()
        ArrowMesh.rotate(angle, uAxis[0], uAxis[1], uAxis[2])
        ArrowMesh.translate(*pos)
        sphereMesh.translate(*pos)
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    track3D(np.array([0, 0, 0]))
