import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
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
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)


class Custom3DAxis(gl.GLAxisItem):
    """Class defined to extend 'gl.GLAxisItem'."""

    def __init__(self, parent, color=(1, 2, 3, 4)):
        gl.GLAxisItem.__init__(self)
        self.parent = parent
        self.c = color

    def add_labels(self):
        """Adds axes labels."""
        x, y, z = self.size()
        # X label
        self.xLabel = CustomTextItem(X=x + 1, Y=-y / 10, Z=-z / 10, text="X(cm)")
        self.xLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.xLabel)
        # Y label
        self.yLabel = CustomTextItem(X=-x / 10, Y=y + 1, Z=-z / 10, text="Y(cm)")
        self.yLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.yLabel)
        # Z label
        self.zLabel = CustomTextItem(X=-x / 10, Y=-y / 10, Z=z + 1, text="Z(cm)")
        self.zLabel.setGLViewWidget(self.parent)
        self.parent.addItem(self.zLabel)

    def add_tick_values(self, xticks=[], yticks=[], zticks=[]):
        """Adds ticks values."""
        x, y, z = self.size()
        print(x, y, z)
        xtpos = np.linspace(0, x, len(xticks))
        ytpos = np.linspace(0, y, len(yticks))
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
            val = CustomTextItem(X=0, Y=0, Z=ztpos[i], text=str(zt))
            val.setGLViewWidget(self.parent)
            self.parent.addItem(val)

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
        ogl.glVertex3f(0, 0, 0)
        ogl.glVertex3f(0, y, 0)
        # Draw X
        ogl.glColor4f(0, 0, 1, 10.6)  # x is blue
        ogl.glVertex3f(0, 0, 0)
        ogl.glVertex3f(x, 0, 0)
        ogl.glEnd()


def track3D(state):
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.resize(800, 600)
    # instance of Custom3DAxis
    axis = Custom3DAxis(w, color=(0.6, 0.6, 0.2, .6))
    axis.setSize(x=40, y=40, z=80)
    # Add axes labels
    axis.add_labels()
    # Add axes tick values
    axis.add_tick_values(xticks=[0, 10, 20, 30, 40], yticks=[0, 10, 20, 30, 40], zticks=[0, 20, 40, 60, 80])
    w.addItem(axis)
    w.opts['distance'] = 100
    # add xy grid
    gx = gl.GLGridItem()
    gx.setSize(x=100, y=100, z=10)
    gx.setSpacing(x=10, y=10)
    w.addItem(gx)

    pos = np.array([[0, 10, 55], [0, -10, 55]])
    plt0 = gl.GLLinePlotItem(pos=pos, width=1)
    w.addItem(plt0)
    plt = gl.GLLinePlotItem(pos=pos, width=2, color=(1, 0, 0, .6))
    w.addItem(plt)
    w.show()
    w.setWindowTitle('3d trajectory')

    i = 1
    pts = pos

    def update():
        nonlocal i, pts, state
        pt = np.array([state[0] * 100, state[1] * 100, state[2] * 100]).reshape(1, 3)
        if pts.size < 150:
            pts = np.concatenate((pts, pt))
        else:
            pts = np.concatenate((pts[-50:, :], pt))
        plt.setData(pos=pts)
        i += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    track3D(np.array([0, 0, 0]))
