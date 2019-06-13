import numpy as np
import math
import time
import threading
import helper
import datetime
import keras
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.backend import clear_session
import tensorflow as tf


class Controller_PID_Point2Point():
    def __init__(self, get_state, get_time, actuate_motors, Motor_limits, Tilt_limits, Yaw_Control_Limits, Z_XY_offset,
                Linear_PID, Linear_To_Angular_Scaler, Yaw_Rate_Scaler, Angular_PID):
        self.actuate_motors = actuate_motors
        self.get_state = get_state
        self.get_time = get_time
        self.MOTOR_LIMITS = Motor_limits
        self.TILT_LIMITS = [(Tilt_limits[0]/180.0)*3.14,(Tilt_limits[1]/180.0)*3.14]
        self.YAW_CONTROL_LIMITS = Yaw_Control_Limits
        self.Z_LIMITS = [self.MOTOR_LIMITS[0] + Z_XY_offset, self.MOTOR_LIMITS[1] - Z_XY_offset]
        self.LINEAR_P = Linear_PID['P']
        self.LINEAR_I = Linear_PID['I']
        self.LINEAR_D = Linear_PID['D']
        self.LINEAR_TO_ANGULAR_SCALER = Linear_To_Angular_Scaler
        self.YAW_RATE_SCALER = Yaw_Rate_Scaler
        self.ANGULAR_P = Angular_PID['P']
        self.ANGULAR_I = Angular_PID['I']
        self.ANGULAR_D = Angular_PID['D']
        self.xi_term = 0
        self.yi_term = 0
        self.zi_term = 0
        self.thetai_term = 0
        self.phii_term = 0
        self.gammai_term = 0
        self.thread_object = None
        self.target = [0,0,0]
        self.yaw_target = 0.0
        self.run = True

    def update(self):
        [dest_x,dest_y,dest_z] = self.target
        [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.get_state()
        x_error = dest_x-x
        y_error = dest_y-y
        z_error = dest_z-z
        self.xi_term += self.LINEAR_I[0]*x_error
        self.yi_term += self.LINEAR_I[1]*y_error
        self.zi_term += self.LINEAR_I[2]*z_error
        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term
        throttle = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta,dest_phi = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1]),np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        gamma_dot_error = (self.YAW_RATE_SCALER* helper.wrap_angle(dest_gamma-gamma)) - gamma_dot
        self.thetai_term += self.ANGULAR_I[0]*theta_error
        self.phii_term += self.ANGULAR_I[1]*phi_error
        self.gammai_term += self.ANGULAR_I[2]*gamma_dot_error
        x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2]*(gamma_dot_error) + self.gammai_term
        z_val = np.clip(z_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        M = np.clip([m1,m2,m3,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        self.actuate_motors(M)
        # print("motors actuated!", self.target)


    def update_target(self,target):
        self.target = target

    def update_yaw_target(self,target):
        self.yaw_target = helper.wrap_angle(target)

    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

class Controller_NeuralNet_Angle_2D():
    def __init__(self, quad, dt, get_state, set_state, get_time, actuate_motors, memory_length, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, ANGLE_DISCRITIZATION, MOVEMENT_LENGTH, will_save):
        #self.ctrl = Controller_PID_Point2Point(get_state, get_time, actuate_motors, Motor_limits, Tilt_limits, Yaw_Control_Limits, Z_XY_offset, Linear_PID, Linear_To_Angular_Scaler, Yaw_Rate_Scaler, Angular_PID)

        self.quad = quad
        self.dt = dt
        self.set_state = set_state

        self.actuate_motors = actuate_motors
        self.get_time = get_time
        self.get_state = get_state

        self.run = True


        # Deep learning stuff

        self.memory = deque(maxlen=2000)

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate


        # number of splits within 360 degrees
        self.ANGLE_DISCRITIZATION = ANGLE_DISCRITIZATION
        self.MOVEMENT_LENGTH = MOVEMENT_LENGTH

        #self.state_size = 12
        self.state_size = 2
        self.action_size = self.ANGLE_DISCRITIZATION
        self.model = self._build_model()
        self.graph = tf.get_default_graph()

        self.x_min = -6
        self.x_max = 6
        self.y_min = -6
        self.y_max = 6
        self.z_min = -6
        self.z_max = 6

        self.will_save = will_save

    def _build_model(self):
        model = Sequential()
        
        model.add(Dense(200, input_dim = self.state_size, activation='relu'))
        #model.add(Dense(300, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #model._make_predict_function()

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):

        currState = self.get_state()[:2]
        x = currState[0]
        y = currState[1]
        z = 0
    
        movement_angle = None

        # With epsilon chance, it will randomly explore
        if np.random.rand() <= self.epsilon:
            movement_angle = random.randrange(self.ANGLE_DISCRITIZATION)
        else:
            with self.graph.as_default():
                output_values = self.model.predict(np.reshape(np.array(currState), [1, self.state_size]))[0]
                movement_angle = np.argmax(output_values)

        x_adjustment = math.cos(math.radians((360/self.ANGLE_DISCRITIZATION) * movement_angle)) * self.MOVEMENT_LENGTH
        y_adjustment = math.sin(math.radians((360/self.ANGLE_DISCRITIZATION) * movement_angle)) * self.MOVEMENT_LENGTH
        self.set_state([x+x_adjustment,y+y_adjustment,0], [0,0,0], 0.3, 0.1, [10,4.5], 1.2)

        finalState = np.array([x+x_adjustment, y+y_adjustment])

        return np.array([np.array(currState), movement_angle, finalState])


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                # Have to make so it gets argmax for each motor seperately and sum to get Q-value for actual state
                with self.graph.as_default():
                    next_state_reshape = np.reshape(next_state, [1, self.state_size])
                    output_values_next_state = self.model.predict(next_state_reshape)[0]
                    target = (reward + self.gamma * np.argmax(output_values_next_state))



            with self.graph.as_default():
                state_reshape = np.reshape(state, [1, self.state_size])
                output_values_state = self.model.predict(state_reshape)[0]

            # action[n] represents the discritized value starting at 0 up to MOTOR_DISCRITIZATION for the nth motor
            # Target should ideally be equal to attempt. So we will use gradient descent to gradually correct it
            attempt = output_values_state[action]

            # This is the Loss we will propogate backwards
            difference = target - attempt

            output_values_state[action] += difference

            # Output from state is tweaked and sent back in
            with self.graph.as_default():
                state_reshape = np.reshape(state, [1, self.state_size])
                output_values_state_reshape = np.reshape(output_values_state, [1, self.action_size])
                self.model.fit(state_reshape, output_values_state_reshape, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def thread_run_graphics(self,update_rate,time_scaling, path):
        update_rate = update_rate*time_scaling
        last_update = self.get_time()


        batch_size = 32
        max_distance = 6
        count = 0
        outer_count = 0
        low_epsilon = False
        dead_count = 0
        total_reward = 0
        inner_range = 5000

        while(self.run==True):

            if outer_count % 10 == 0:
                print("==================")
                print("After ", 10*inner_range, " iterations,")
                print(dead_count, " have died an honorable death in the name of science")
                print(total_reward, "total reward accumulated")
                print("epsilon is now ", self.epsilon)
                print("==================")
                dead_count = 0
                total_reward = 0


            temp_reward = 0
            temp_gamma = 1

            for t in range(inner_range):
                time.sleep(0)
                self.time = self.get_time()
                if (self.time - last_update).total_seconds() > update_rate:
                    state_action_state = self.update()
                    # Reward is simply how much closer it is to the origin

                    done = math.sqrt(state_action_state[2][0]**2 + state_action_state[2][1]**2) > max_distance

                    change_in_distance = (state_action_state[0][0]**2 + state_action_state[0][1]**2) - (state_action_state[2][0]**2 + state_action_state[2][1]**2 )
                    reward = change_in_distance if not done else -2

                    reward *= 10

                    temp_reward += temp_gamma*reward
                    temp_gamma *= self.gamma

                    if count % 20000 == 0:
                        if self.will_save:
                            print("Saved!", count)
                            self.save(path)
                        count = 0

                    count += 1

                    self.remember(state_action_state[0], state_action_state[1], reward, state_action_state[2], done)

                    if done:
                        temp_gamma = 1
                        print("Run had ", temp_reward, "reward")
                        total_reward += temp_reward
                        temp_reward = 0

                        dead_count += 1
                        x = random.randrange(-max_distance, max_distance)
                        y = random.randrange(-max_distance, max_distance)
                        z = 0
                        self.set_state([x,y,z], [0,0,0], 0.3, 0.1, [10,4.5], 1.2)


                    last_update = self.time


            outer_count += 1
            if len(self.memory) > batch_size and self.will_save:
                self.replay(batch_size)

    def load(self, name):
        try:
            with self.graph.as_default():
                self.model.load_weights(name)
            print("Loaded!")
        except:
            print("Warning: model not found. Creating new model")

    def save(self, name):
        with self.graph.as_default():
            self.model.save_weights(name)

    def start_thread_graphics(self,path, update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run_graphics,args=(update_rate,time_scaling, path))
        self.load(path)
        self.thread_object.start()

    def start_thread_no_graphics(self,path, update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run_no_graphics,args=(update_rate,time_scaling, path))
        self.load(path)
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

class Controller_PID_Velocity(Controller_PID_Point2Point):
    def update(self):
        [dest_x,dest_y,dest_z] = self.target
        [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.get_state()
        x_error = dest_x-x_dot
        y_error = dest_y-y_dot
        z_error = dest_z-z
        self.xi_term += self.LINEAR_I[0]*x_error
        self.yi_term += self.LINEAR_I[1]*y_error
        self.zi_term += self.LINEAR_I[2]*z_error
        dest_x_dot = self.LINEAR_P[0]*(x_error) + self.LINEAR_D[0]*(-x_dot) + self.xi_term
        dest_y_dot = self.LINEAR_P[1]*(y_error) + self.LINEAR_D[1]*(-y_dot) + self.yi_term
        dest_z_dot = self.LINEAR_P[2]*(z_error) + self.LINEAR_D[2]*(-z_dot) + self.zi_term
        throttle = np.clip(dest_z_dot,self.Z_LIMITS[0],self.Z_LIMITS[1])
        dest_theta = self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_x_dot*math.sin(gamma)-dest_y_dot*math.cos(gamma))
        dest_phi = self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_x_dot*math.cos(gamma)+dest_y_dot*math.sin(gamma))
        dest_gamma = self.yaw_target
        dest_theta,dest_phi = np.clip(dest_theta,self.TILT_LIMITS[0],self.TILT_LIMITS[1]),np.clip(dest_phi,self.TILT_LIMITS[0],self.TILT_LIMITS[1])
        theta_error = dest_theta-theta
        phi_error = dest_phi-phi
        gamma_dot_error = (self.YAW_RATE_SCALER*helper.wrap_angle(dest_gamma-gamma)) - gamma_dot
        self.thetai_term += self.ANGULAR_I[0]*theta_error
        self.phii_term += self.ANGULAR_I[1]*phi_error
        self.gammai_term += self.ANGULAR_I[2]*gamma_dot_error
        x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
        y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
        z_val = self.ANGULAR_P[2]*(gamma_dot_error) + self.gammai_term
        z_val = np.clip(z_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
        m1 = throttle + x_val + z_val
        m2 = throttle + y_val - z_val
        m3 = throttle - x_val + z_val
        m4 = throttle - y_val - z_val
        M = np.clip([m1,m2,m3,m4],self.MOTOR_LIMITS[0],self.MOTOR_LIMITS[1])
        self.actuate_motors(M)
