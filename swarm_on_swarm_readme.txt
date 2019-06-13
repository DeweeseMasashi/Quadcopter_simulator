================================
Alex's notes (documentation)
================================
*) To add a new simulation, add a new conditional statement at bottom of quad_sim.py
*) self.ax.plot([0,2],[0,2],[0,5]). these lists define a 3d line. the first two determine the line on the x-y plane and then the last list determines the heights of the first two points
*) Vehicle is actually created when you call, say 
    self.quads[key]['l1'], = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)
    - This is because the thing that ax.plot returns is an object which lets you edit its coordinates. When it's editted, the plot also updates and changes its coordinates
    - Specifically, we can set coords of the "vehicle" returned by plot by using set_data to change the x-y coordinates and using set_3d_properties to change the height

================================
Alex's notes (Refactoring)
================================
*) currently, quadcopters are represented as a dictionary. I'll change it to make quadcopter objects so we can eventually utilize inheritence for general vehicles
*) It looks like the old quadcopter object did stuff with multithreading working with multiple quadcopters. To keep this functionality, I created a QuadManager class which holds information pertaining to a list of quadcopters
*) I made this note in the update function for quadcopter.py
        NOTE: This is code from before refactor. Notice how set_f_params takes in a key.
        What was going on was it initialized one self.ode for every quadcopter and
        just used set_f_params with a key input. I wonder if this is faster than
        what we have now...

TODO Others:
*) looks like there are many functions that are repeated like rotation_matrix. Maybe create a file dedicated to those types of helper functions?
*) refactor controller

TODO Myself:
*) finish refactoring controller_pid_velocity

================================
Yifan's notes (Refactoring) 04/25/2019
================================

*) I created a new py file that defines the general helper functions, so things like rotation_maxtrix, wrap_angle do not repeat anymore.
*) I removed comments that's already finished refactoring, leaving important comments such as params and Controller_PID_Velocity unchanged.
*) Refactored Controller_PID_Point2Point.

================================
Alex's notes (DQN)
================================
*) To start off run:
    python quad_sim.py --sim nn_angle_test_2d
*) If you have keras and other python libraries installed it should work and
    you should see a drone jittering near the center
*) To change architecture, as of now, controller.py must be modified
*) hyper parameters can be changed from quad_sim.py
*) To experiment with training, go to quad_sim.py under the 2D Demo and uncomment the code 
    that initializes the controller. This will begin a new training session starting
    from scratch. BE SURE TO CHANGE THE FILE NAME OR IT WILL OVER RIDE THE INCLUDED DEMO
