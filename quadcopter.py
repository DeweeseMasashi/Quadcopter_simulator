import numpy as np
import math
import scipy.integrate
import time
import datetime
import threading
import helper

class Propeller():
    """ This class mainly calculates the behavior(thrust) of a Propeller

        Instance Attributes
        -------------------
        self.dia  : (inches) propeller diameter
        self.pitch : (inches) propeller pitch
        self.thrust_unit : (string) the unit that we want for the resulting thrust calculated
        self.speed : (rotations/min) how fast is the propeller spinning
        self.thrust: (in thrust_unit) the result we would like to present

        Methods
        -------
            __init__  : Constructor
            set_speed : Calculate the thrust
    """

    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):

        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0 #RPM
        self.thrust = 0

    def set_speed(self,speed):
        """ The main calculation for thrust

            From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        """
        self.speed = speed
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia,3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust*0.101972

# This manages the meta data for a list of quadcopters
class QuadManager():

    def __init__(self, quad_list):
        self.quad_list = quad_list
        self.thread_object = None
        self.time = datetime.datetime.now()
        self.run = True


    def thread_run(self,dt,time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while(self.run==True):
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                #self.update(dt)
                for q in self.quad_list:
                    q.update(dt)
                last_update = self.time


    def start_thread(self,dt=0.002,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(dt,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def get_time(self):
        return self.time

# This manages the meta data for a list of quadcopters
class QuadManagerTimeOnly():

    def __init__(self):
        self.thread_object = None
        self.time = datetime.datetime.now()
        self.run = True


    def thread_run(self,dt,time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while(self.run==True):
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                #self.update(dt)
                last_update = self.time


    def start_thread(self,dt=0.002,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(dt,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    def get_time(self):
        return self.time

class Quadcopter():

    """ The Quadcopter class

        From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky

        Instance Attributes
        -------------------
        self.ode : 
        self.position : (list of length) x, y, z
        self.orientation : (list of angle) theta, phi, gamma
        self.L : (length) length of the arm
        self.r : (length) radius of a sphere representing the center blob of the quadcopter
        self.prop_size : (list of 2 elements) [propeller_diameter, propeller_pitch]
        self.weight : weight of the quadcopter
        self.gravity : gravity on the planet
        self.b : 
        self.state : state space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
        self.m* : individual propellers that the quadcopter has
        self.I : the moment of inertia matrix
        self.invI : the inverse of self.I

        Methods
        -------
        __init__ : Constructor
        update : update the state of the quadcopter 
        set_motor_speeds : takes in a list of speeds and assign it to each self.m* respectively
        state_dot : take the derivative of the state vector
        get_position : get [x, y, z]
        get_linear_rate : get [x_dot, y_dot, z_dot]
        get_orientation : get [theta, phi, gamma]
        get_angular_rate : get [theta_dot, phi_dot, gamma_dot]
        get_state : get self.state
        set_position : set the position of the quadcopter
        set_orientation : set the orientation of the quadcopter
    """
    
    def __init__(self,position, orientation, L, r, prop_size, weight, gravity=9.81,b=0.0245):
    #def __init__(self,quads,gravity=9.81,b=0.0245):
        #self.quads = quads
        self.ode =  scipy.integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')
        self.position = position
        self.orientation = orientation
        self.L = L
        self.r = r
        self.prop_size = prop_size
        self.weight = weight
        self.g = gravity
        self.b = b

        self.state = np.zeros(12)
        self.state[0:3] = self.position
        self.state[6:9] = self.orientation
        self.m1 = Propeller(self.prop_size[0],self.prop_size[1])
        self.m2 = Propeller(self.prop_size[0],self.prop_size[1])
        self.m3 = Propeller(self.prop_size[0],self.prop_size[1])
        self.m4 = Propeller(self.prop_size[0],self.prop_size[1])
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx=((2*self.weight*r**2)/5)+(2*self.weight*L**2)
        iyy=ixx
        izz=((2*self.weight*r**2)/5)+(4*self.weight*L**2)
        self.I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.invI = np.linalg.inv(self.I)



    def update(self, dt):
        """ The update to the state is performed by an ODE solver from the current state to a new state over a period of dt time(defined by user). 
        
        It uses the vode ODE solver available from the SciPy library. 
        It has an update method to update the state, which is run on a thread at intervals defined by the time scaling factor. 
        The thread can be started by the start_thread method.

        NOTE: This is code from before refactor. Notice how set_f_params takes in a key.
        What was going on was it initialized one self.ode for every quadcopter and
        just used set_f_params with a key input. I wonder if this is faster than
        what we have now...

        for key in self.quads:
            self.ode.set_initial_value(self.quads[key]['state'],0).set_f_params(key)
            self.quads[key]['state'] = self.ode.integrate(self.ode.t + dt)
            self.quads[key]['state'][6:9] = self.wrap_angle(self.quads[key]['state'][6:9])
            self.quads[key]['state'][2] = max(0,self.quads[key]['state'][2])
        """
        self.ode.set_initial_value(self.state,0).set_f_params()
        self.state = self.ode.integrate(self.ode.t + dt)
        self.state[6:9] = helper.wrap_angle(self.state[6:9])
        self.state[2] = max(0,self.state[2])

    def set_motor_speeds(self,speeds):
        self.m1.set_speed(speeds[0])
        self.m2.set_speed(speeds[1])
        self.m3.set_speed(speeds[2])
        self.m4.set_speed(speeds[3])

    def state_dot(self, time, state):
        state_dot = np.zeros(12)
        # The velocities(t+1 x_dots equal the t x_dots)
        state_dot[0] = self.state[3]
        state_dot[1] = self.state[4]
        state_dot[2] = self.state[5]
        # The acceleration
        x_dotdot = np.array([0,0,-self.weight*self.g]) + np.dot(helper.rotation_matrix(self.state[6:9]),np.array([0,0,(self.m1.thrust + self.m2.thrust + self.m3.thrust + self.m4.thrust)]))/self.weight
        state_dot[3] = x_dotdot[0]
        state_dot[4] = x_dotdot[1]
        state_dot[5] = x_dotdot[2]
        # The angular rates(t+1 theta_dots equal the t theta_dots)
        state_dot[6] = self.state[9]
        state_dot[7] = self.state[10]
        state_dot[8] = self.state[11]
        # The angular accelerations
        omega = self.state[9:12]
        tau = np.array([self.L*(self.m1.thrust-self.m3.thrust), self.L*(self.m2.thrust-self.m4.thrust), self.b*(self.m1.thrust-self.m2.thrust+self.m3.thrust-self.m4.thrust)])
        omega_dot = np.dot(self.invI, (tau - np.cross(omega, np.dot(self.I,omega))))
        state_dot[9] = omega_dot[0]
        state_dot[10] = omega_dot[1]
        state_dot[11] = omega_dot[2]
        return state_dot


    def get_position(self):
        return self.state[0:3]

    def get_linear_rate(self):
        return self.state[3:6]

    def get_orientation(self):
        return self.state[6:9]

    def get_angular_rate(self):
        return self.state[9:12]

    def get_state(self):
        return self.state

    def set_state(self, position, orientation, L, r, prop_size, weight, gravity=9.81,b=0.0245):
        self.ode =  scipy.integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')
        self.position = position
        self.orientation = orientation
        self.L = L
        self.r = r
        self.prop_size = prop_size
        self.weight = weight
        self.g = gravity
        self.b = b

        self.state = np.zeros(12)
        self.state[0:3] = self.position
        self.state[6:9] = self.orientation
        self.m1 = Propeller(self.prop_size[0],self.prop_size[1])
        self.m2 = Propeller(self.prop_size[0],self.prop_size[1])
        self.m3 = Propeller(self.prop_size[0],self.prop_size[1])
        self.m4 = Propeller(self.prop_size[0],self.prop_size[1])
        # From Quadrotor Dynamics and Control by Randal Beard
        ixx=((2*self.weight*r**2)/5)+(2*self.weight*L**2)
        iyy=ixx
        izz=((2*self.weight*r**2)/5)+(4*self.weight*L**2)
        self.I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self.invI = np.linalg.inv(self.I)


    def set_position(self, position):
        self.state[0:3] = position

    def set_orientation(self, orientation):
        self.state[6:9] = orientation
