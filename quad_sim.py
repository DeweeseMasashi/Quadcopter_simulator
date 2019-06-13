import quadcopter,gui,controller
import signal
import sys
import argparse
import datetime

# Constants
TIME_SCALING = 1.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
#QUAD_DYNAMICS_UPDATE = 0.002 # seconds
QUAD_DYNAMICS_UPDATE = 0.002 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.0005 # seconds
run = True


# Added for testing to make sure refactor works
def TwoQuadTest():
    # Set goals to go to
    GOALS_1 = [(-1,-1,4),(1,1,2)]
    GOALS_2 = [(1,-1,2),(-1,1,4)]
    # Define the quadcopters
    # def __init__(self,position, orientation, L, r, prop_size, weight, gravity=9.81,b=0.0245):
    quad_list = [quadcopter.Quadcopter([1,0,4], [0,0,0], 0.3, 0.1, [10,4.5], 1.2), quadcopter.Quadcopter([-1,0,4], [0,0,0], 0.15, 0.05, [6,4.5], 0.7)]
    # Controller parameters
    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controllers
    gui_object = gui.GUI(quads=quad_list)
    #quad = quadcopter.Quadcopter(quads=QUADCOPTERS)

    quad_manager = quadcopter.QuadManager(quad_list)

    ctrl1 = controller.Controller_PID_Point2Point(quad_list[0].get_state,quad_manager.get_time,quad_list[0].set_motor_speeds,
                                                   [4000,9000], [-10,10], [-900,900], 500, {'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                                                   [1,1,0], 0.18, {'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]})
    ctrl2 = controller.Controller_PID_Point2Point(quad_list[1].get_state,quad_manager.get_time,quad_list[1].set_motor_speeds,
                                                    [4000,9000], [-10,10], [-900,900], 500, {'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                                                   [1,1,0], 0.18, {'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]})

    # Start the threads
    quad_manager.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl1.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl2.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    while(run==True):
        for goal1,goal2 in zip(GOALS_1,GOALS_2):
            ctrl1.update_target(goal1)
            ctrl2.update_target(goal2)
            for i in range(150):
                for q in gui_object.quads:
                    q.position = q.get_position()
                    q.orientation = q.get_orientation()
                gui_object.update()
    quad_manager.stop_thread()
    ctrl1.stop_thread()
    ctrl2.stop_thread()

# I took out quad manager and gave that info to NN controller so it can update drones when it wants to

def PIDExperiment():
    # Set goals to go to
    GOALS_1 = [(0,-0.5,0),(0,0.5,0)]
    # Define the quadcopters
    quad_list = [quadcopter.Quadcopter([0,0,0], [0,0,0], 0.3, 0.1, [10,4.5], 1.2)]
    # Controller parameters

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controllers
    gui_object = gui.GUI(quads=quad_list)
    #quad = quadcopter.Quadcopter(quads=QUADCOPTERS)

    quad_manager = quadcopter.QuadManager(quad_list)

    ctrl1 = controller.Controller_PID_Velocity(quad_list[0].get_state,quad_manager.get_time,quad_list[0].set_motor_speeds,
                                                   [4000,9000], [-10,10], [-900,900], 500, {'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                                                   [1,1,0], 0.18, {'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]})

    # Start the threads
    quad_manager.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl1.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # ctrl2.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions

    update_rate = 0.01;
    while(run==True):
        for goal1 in GOALS_1:
            ctrl1.update_target(goal1)
            last_update = datetime.datetime.now()
            #ctrl2.update_target(goal2)
            #for i in range(10):
            while((datetime.datetime.now() - last_update).total_seconds() < update_rate):
                for q in gui_object.quads:
                    q.position = q.get_position()
                    q.orientation = q.get_orientation()
                gui_object.update()
    quad_manager.stop_thread()
    ctrl1.stop_thread()


def NNQuadAngleTest2D():

    # Define the quadcopters
    quad_list = [quadcopter.Quadcopter([1,0,0], [0,0,0], 0.3, 0.1, [10,4.5], 1.2)]

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)

    # Make objects for quadcopter, gui and controllers
    gui_object = gui.GUI_Neural(quads=quad_list)

    quad_manager = quadcopter.QuadManagerTimeOnly()

    #def __init__(self, quad, dt, get_state, set_state, get_time, actuate_motors, memory_length, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, ANGLE_DISCRITIZATION, MOVEMENT_LENGTH, will_save):
    ctrl1 = controller.Controller_NeuralNet_Angle_2D(quad_list[0],5, quad_list[0].get_state,quad_list[0].set_state,quad_manager.get_time, quad_list[0].set_motor_speeds, 2000, 0.95, 0, 0.995, 0, 0.001, 36, 0.1, False)
    #ctrl1 = controller.Controller_NeuralNet_Angle_2D(quad_list[0],5, quad_list[0].get_state,quad_list[0].set_state,quad_manager.get_time, quad_list[0].set_motor_speeds, 2000, 0.95, 1, 0.995, 0.2, 0.001, 36, 0.1, True)

    # Start the threads
    quad_manager.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl1.start_thread_graphics(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING, path = 'models/nn_angle_test/angle_demo.hdf5')

    while(run==True):
        for q in gui_object.quads:
            q.position = q.get_position()
            q.orientation = q.get_orientation()
        gui_object.update()

    quad_manager.stop_thread()
    ctrl1.stop_thread()


def parse_args():
    parser = argparse.ArgumentParser(description="Quadcopter Simulator")
    parser.add_argument("--sim", help='single_p2p, multi_p2p or single_velocity', default='single_p2p')
    parser.add_argument("--time_scale", type=float, default=-1.0, help='Time scaling factor. 0.0:fastest,1.0:realtime,>1:slow, ex: --time_scale 0.1')
    parser.add_argument("--quad_update_time", type=float, default=0.0, help='delta time for quadcopter dynamics update(seconds), ex: --quad_update_time 0.002')
    parser.add_argument("--controller_update_time", type=float, default=0.0, help='delta time for controller update(seconds), ex: --controller_update_time 0.005')
    return parser.parse_args()

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)

if __name__ == "__main__":
    args = parse_args()
    if args.time_scale>=0: TIME_SCALING = args.time_scale
    if args.quad_update_time>0: QUAD_DYNAMICS_UPDATE = args.quad_update_time
    if args.controller_update_time>0: CONTROLLER_DYNAMICS_UPDATE = args.controller_update_time
    if args.sim == 'two_quad_test':
        TwoQuadTest()
    if args.sim == 'nn_angle_test_2d':
        NNQuadAngleTest2D()
    if args.sim == 'pid_experiment':
        PIDExperiment()
