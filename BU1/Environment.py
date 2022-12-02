import math

import numpy as np

import HC


class Environment(object):
    action_set = []
    state = []
    #dt=0;

    def __init__(self):
        # cart position in the horizontal direction x = state[0], and cart velocity denoted by x_ = state[1]
        # vertical angle of the pole denoted by tetha = state[2], angular velocity of the pole tetha_ = state[3]
        self.x=0
        self.x_list=[]
        self.Time_stamps=0;
        self.state = [0,0,0,0]
        self.action_set = [5,22.5,50]
        self.x=0
        self.N = 100          # number of uniform timesteps to sample in optimization (decrease for computation speed)
        self.x0 = [0,0,0]    # start [pos,vel,acc]
        self.xf = [10,0,0]   # final [pos,vel,acc]
        self.dur = [0,5]     # [t0,tf] start and end time of trajectory
        self.optimizer = HC.MinJerkOptimization(self.x0,self.xf,self.dur,self.N)
        self.optimizer.bias['myopia'], self.optimizer.bias['rushed'] = 0.0, 0.0 # default
        self.J = self.optimizer.get_jerk()
        self.xd = self.optimizer.get_state_trajectory(self.J)
        self.dt=self.optimizer.dt
        self.tt=0
        self.r=0
        self.JJ=0

    def apply_action(self,action,phi):
        j=1
        reward=0
        # while j<10:
        #     u = phi*self.action_set[action]
        #     self.get_current_state(u)
        #     reward =reward-self.get_reward()
        #     j=j+1

        rewards = np.zeros(9)
        for j in range(9):
            u = phi * self.action_set[action]
            self.get_current_state(u)
            rewards[j] = - self.get_reward()
        reward =  np.linalg.norm(rewards)
        # reward = np.mean(rewards)
        return reward, self.state

    def get_state_variable(self,variable_name):

        if variable_name == 'V':
            return self.state[0]
        elif variable_name == "F":
            return self.state[1]
        elif variable_name == "V_":
            return self.state[2]
        else:
            return self.state[3]

    def set_state_variable(self,variable_name,value):

        if variable_name == 'V':
            self.state[0] = value
        elif variable_name == "F":
            self.state[1] = value
        elif variable_name == "V_":
            self.state[2] = value
        elif variable_name == "F_":
            self.state[3] = value

    def get_current_state(self,u):
        M=1
        V__old=self.get_state_variable('V_')
        self.set_state_variable('V_',(1/M)*(self.get_state_variable('F')-u*self.get_state_variable('V')))
        V__new=self.get_state_variable('V_')
        self.JJ=(V__new-V__old)/self.dt
        self.set_state_variable('V', self.get_state_variable('V') + (self.get_state_variable('V_') *self.dt))
        self.x=self.x+self.get_state_variable('V')*self.dt;
        self.x_list.append(self.x)
        stiffness = 4.0 # increase for more aggressive/accurate tracking
        damping = 1   # increase if unstable
        mass = 0        # !!! need to double-check calculations for this !!!!
        Hcontrol = HC.ImpedanceController(stiffness,damping,mass,self.xd[self.Time_stamps],self.dt)
        Hcontrol.tauf = 3
        F_old=self.get_state_variable('F')
        self.set_state_variable('F',Hcontrol.get_force([self.x,self.get_state_variable('V'),self.get_state_variable('V_')],self.xd[self.Time_stamps]))
        F_new=self.get_state_variable('F')
        self.set_state_variable('F_',(F_new-F_old)/(self.dt))
        self.Time_stamps=self.Time_stamps+1;
    def get_reward(self):
            r=self.JJ#*self.JJ
            return r