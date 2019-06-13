import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D
import sys
from quadcopter import  Quadcopter
import helper

# Increases the range of the axes for neural network training
class GUI_Neural():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, quads):
        max_distance = 6
        self.quads = quads
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-max_distance, max_distance])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-max_distance, max_distance])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-max_distance, max_distance])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.init_plot()
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)

    def init_plot(self):
        for q in self.quads:
            q.l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
            q.l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
            q.hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)

    def update(self):
        for q in self.quads:
            R = helper.rotation_matrix(q.orientation)
            L = q.L
            points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
            points = np.dot(R,points)
            points[0,:] += q.position[0]
            points[1,:] += q.position[1]
            points[2,:] += q.position[2]
            q.l1.set_data(points[0,0:2],points[1,0:2])
            q.l1.set_3d_properties(points[2,0:2])
            q.l2.set_data(points[0,2:4],points[1,2:4])
            q.l2.set_3d_properties(points[2,2:4])
            q.hub.set_data(points[0,5],points[1,5])
            q.hub.set_3d_properties(points[2,5])
        plt.pause(0.000000000000001)

    def keypress_routine(self,event):
        sys.stdout.flush()
        if event.key == 'x':
            y = list(self.ax.get_ylim3d())
            y[0] += 0.2
            y[1] += 0.2
            self.ax.set_ylim3d(y)
        elif event.key == 'w':
            y = list(self.ax.get_ylim3d())
            y[0] -= 0.2
            y[1] -= 0.2
            self.ax.set_ylim3d(y)
        elif event.key == 'd':
            x = list(self.ax.get_xlim3d())
            x[0] += 0.2
            x[1] += 0.2
            self.ax.set_xlim3d(x)
        elif event.key == 'a':
            x = list(self.ax.get_xlim3d())
            x[0] -= 0.2
            x[1] -= 0.2
            self.ax.set_xlim3d(x)
class GUI():
    # 'quad_list' is a dictionary of format: quad_list = {'quad_1_name':{'position':quad_1_position,'orientation':quad_1_orientation,'arm_span':quad_1_arm_span}, ...}
    def __init__(self, quads):
        self.quads = quads
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 5.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')
        self.init_plot()
        self.fig.canvas.mpl_connect('key_press_event', self.keypress_routine)

    def init_plot(self):
        for q in self.quads:
            q.l1, = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
            q.l2, = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)
            q.hub, = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)

    def update(self):
        for q in self.quads:
            R = helper.rotation_matrix(q.orientation)
            L = q.L
            points = np.array([ [-L,0,0], [L,0,0], [0,-L,0], [0,L,0], [0,0,0], [0,0,0] ]).T
            points = np.dot(R,points)
            points[0,:] += q.position[0]
            points[1,:] += q.position[1]
            points[2,:] += q.position[2]
            q.l1.set_data(points[0,0:2],points[1,0:2])
            q.l1.set_3d_properties(points[2,0:2])
            q.l2.set_data(points[0,2:4],points[1,2:4])
            q.l2.set_3d_properties(points[2,2:4])
            q.hub.set_data(points[0,5],points[1,5])
            q.hub.set_3d_properties(points[2,5])
        plt.pause(0.000000000000001)

    def keypress_routine(self,event):
        sys.stdout.flush()
        if event.key == 'x':
            y = list(self.ax.get_ylim3d())
            y[0] += 0.2
            y[1] += 0.2
            self.ax.set_ylim3d(y)
        elif event.key == 'w':
            y = list(self.ax.get_ylim3d())
            y[0] -= 0.2
            y[1] -= 0.2
            self.ax.set_ylim3d(y)
        elif event.key == 'd':
            x = list(self.ax.get_xlim3d())
            x[0] += 0.2
            x[1] += 0.2
            self.ax.set_xlim3d(x)
        elif event.key == 'a':
            x = list(self.ax.get_xlim3d())
            x[0] -= 0.2
            x[1] -= 0.2
            self.ax.set_xlim3d(x)
