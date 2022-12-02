import numpy as np

import FuzzySet
import StateVariable
import FQL
import FIS
from Environment import Environment
import matplotlib.pyplot as plt
import sys
import HC
reward=0;
phi=0;
#state_value=[0,0,0,0]
action_list=[];
Comu_reward_list=[];
F_list=[]
F_dot_list=[]
V_list=[]
V_dot_list=[]
Comu_reward=0;
x_RL=[];
# Create FIS
fuzzy_set = {}
fuzzy_set['v'] = {'range':[-2,6],'nbin':4}
fuzzy_set['F'] = {'range':[-8,8],'nbin':4}
fuzzy_set['dvdt'] = {'range':[-5,5],'nbin':4}
fuzzy_set['dFdt'] = {'range':[-6,6],'nbin':4}


xi_tmp = []
for key in fuzzy_set:
    fset = fuzzy_set[key]
    centers = np.linspace(fset['range'][0],fset['range'][1],fset['nbin'])
    size = (fset['range'][1]-fset['range'][0])/(fset['nbin']-1)
    xtmp = StateVariable.InputStateVariable(*[FuzzySet.Triangles(cent-size, cent, cent+size) for cent in centers])
    xi_tmp.append(xtmp)

x1 = xi_tmp[0]
x2 = xi_tmp[1]
x3 = xi_tmp[2]
x4 = xi_tmp[3]
# x1 = StateVariable.InputStateVariable(FuzzySet.Triangles(-1,0,1), FuzzySet.Triangles(0,1,2), FuzzySet.Triangles(1,2,3),FuzzySet.Triangles(2,3,4),FuzzySet.Triangles(3,4,5))
# x2 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3,-2,-1), FuzzySet.Triangles(-2,-1,0), FuzzySet.Triangles(-1,0,1),FuzzySet.Triangles(0,1,2),FuzzySet.Triangles(1,2,3))
# x3 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3,-2,-1), FuzzySet.Triangles(-2,-1,0), FuzzySet.Triangles(-1,0,1),FuzzySet.Triangles(0,1,2),FuzzySet.Triangles(1,2,3))
# x4 = StateVariable.InputStateVariable(FuzzySet.Triangles(-3,-2,-1), FuzzySet.Triangles(-2,-1,0), FuzzySet.Triangles(-1,0,1),FuzzySet.Triangles(0,1,2),FuzzySet.Triangles(1,2,3))
fis = FIS.Build(x1,x2,x3,x4)


# Create Model
x_list = []
model = FQL.Model(gamma = 0.6,#gamma = 0.95,
                  alpha = 0.05 ,
                  ee_rate = 0.9,
                  q_initial_value = 'random',
                  action_set_length = 3, fis = fis)
env = Environment()
env.__init__()
max=-sys.maxsize
# Comu_reward=-1000000
# xtol = np.array([0.1,0.1,0.1])
# max_iter =int(2*env.N/10)
# nEpisodes = 50

#
stats = {}
stats['epi_reward'] = []
stats['epi_len'] = []
#
# for epi in range(nEpisodes):
#
#     stats['epi_reward'].append(0)
#     stats['epi_len'].append(0)
#     Comu_reward_list.append(Comu_reward)
#     # if Comu_reward > max:
#     #     max = Comu_reward
#     #     x_RL = env.x_list;
#     x_list = []
#
#     env.__init__()
#     action, phi = model.get_initial_action(env.state)
#     action_list.append(action)
#     reward, state_value = env.apply_action(action, phi)
#
#     for iter in range(max_iter): #max_iter
#         print(f'\rProg=[epi:{epi} / {nEpisodes}; iter={iter} / {max_iter}; ]',end='')
#         action, phi = model.run(state_value, reward)
#         reward, state_value = env.apply_action(action, phi)
#         stats['epi_reward'][-1] += reward
#         stats['epi_len'][-1] = iter
#
#         F_dot_list.append(env.get_state_variable('F_'))
#         V_dot_list.append(env.get_state_variable('V_'))
#         F_list.append(env.get_state_variable('F'))
#         V_list.append(env.get_state_variable('V'))
#
#         Xcurr = np.array([env.x,env.state[0],env.state[2]])
#         print(f'X={Xcurr}', end='')
#
#         if np.all(env.xd[-1,:] - Xcurr < xtol):
#             print(f'####### Target Reached ########')
#             break
#
# x_RL = env.x_list



max_iter = 10
for iteration in range (0,max_iter):
    try:
        Rcum = stats['epi_reward'][-1]

        print(f'\rProg=[{100*iteration/max_iter}%;JJ={env.JJ} Repi = {Rcum}]',end='')
    except:
        pass

    if iteration % (2*env.N/10) == 0:
        # Comu_reward_list.append(Comu_reward)
        # if Comu_reward>max:
        #     max=Comu_reward
        #     x_RL=env.x_list
        x_list=[]
        F_list = []
        F_dot_list = []
        V_list = []
        V_dot_list = []
        action_list = []

        stats['epi_reward'].append(0)
        stats['epi_len'].append(0)

        # Comu_reward=0
        env.__init__()
        action, phi = model.get_initial_action(env.state)
        reward, state_value = env.apply_action(action,phi)


    action, phi = model.run(state_value, reward)

    action_list.append(phi*env.action_set[int(action)])
    reward, state_value = env.apply_action(action,phi)
    print(env.get_reward())
    # Comu_reward=Comu_reward+reward
    stats['epi_reward'][-1] += reward
    stats['epi_len'][-1] = iteration
    F_dot_list.append(env.get_state_variable('F_'))
    V_dot_list.append(env.get_state_variable('V_'))
    F_list.append(env.get_state_variable('F'))
    V_list.append(env.get_state_variable('V'))

# trajectory/ without RL Agent
stiffness = 4.0 # increase for more aggressive/accurate tracking
damping = 1   # increase if unstable
mass = 0        # !!! need to double-check calculations for this !!!!
Hcontrol = HC.ImpedanceController(stiffness,damping,mass,env.xd,env.dt)
Hcontrol.tauf = 3
xH = Hcontrol.get_state_trajectory(env.xd)

plt.figure(figsize=(14,3))
plt.plot(action_list)
plt.title('Action')
# plt.show()


plt.figure(figsize=(14,3))
plt.plot(Comu_reward_list)
# plt.plot(stats['epi_reward'])
plt.ylabel('reward')
plt.xlabel('episode')
# plt.show()


damping = np.arange(100)
episode_reward = np.arange(100)



nRows,nCols = 3,1
fig,axs = plt.subplots(3,1)
axs = np.array(axs).reshape([nRows,nCols])

# Reward Plot
axs[0,0].plot(episode_reward)
axs[0,0].set_xlabel('Episode')
axs[0,0].set_ylabel('Reward')
axs[0,0].set_title('Cumulative Episode Rewards')


# Position Plot
axs[1,0].plot(x_RL)
axs[1,0].plot(xH[0:len(x_RL),0,0])
axs[1,0].legend(['RL','Human'])
axs[1,0].set_xlabel('Time (t)')
axs[1,0].set_ylabel('Position')
axs[1,0].set_title('Final Motion Behaviors')


# Dampings
axs[2,0].plot(damping)
axs[2,0].set_xlabel('Time (t)')
axs[2,0].set_ylabel('Damping ($U_t$)')
axs[2,0].set_title('Robot Action (Damping)')

# Draw plot
plt.tight_layout()
plt.show()









