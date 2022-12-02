import numpy as np

from FIS import FuzzyInferenceSystem
from Environment import Environment
import matplotlib.pyplot as plt

FQL_settings = {}
FQL_settings['num_episodes'] = 10       # number of repeated interactions
FQL_settings['max_epi_duration'] = 10   # terminal duration in seconds
FQL_settings['dtau'] = 0.01             # period of control loop sampling
FQL_settings['tau2t'] = 10              # num of tau samples per 1 t sample
FQL_settings['tau_iterable'] = np.arange(0, FQL_settings['max_epi_duration'], FQL_settings['dtau'])
FQL_settings['dt'] = FQL_settings['dtau']*FQL_settings['tau2t'] # period of control loop sampling


FIS_settings = {
    'v':    {'range': [-2, 6], 'nbin': 4},
    'F':    {'range': [-8, 8], 'nbin': 4},
    'dvdt': {'range': [-5, 5], 'nbin': 4},
    'dFdt': {'range': [-6, 6], 'nbin': 4}
}


env_params = {
    'desired_pos':     5.,
    'timestep':        FQL_settings['dtau'],
    'mass':            1.3,
    'K_human':         10.0,
    'D_human':         5.0,
    'error_thresh':    0.01,
    'max_duration':    10.,
    'start_pos':       0.,
    'actions':         [10, 22.5, 50]
}



FIS = FuzzyInferenceSystem()
FIS.add_fuzzy_set('v', [-10, 10], 3)
FIS.add_fuzzy_set('a', [-5, 5], 3)
FIS.add_fuzzy_set('F', [-5, 5], 3)
FIS.add_fuzzy_set('dFdt', [-5, 5], 3)
FIS.generate_rules()




class FuzzyHandler(object):
    def __init__(self,alpha=0.05,gamma=0.8):
        self.q = np.zeros(5,5)
        self.ax_action = 1
        self.Si = np.arange(FIS.nRules) # list of all rule indicies
        self.A = env_params['actions']# discrete crisp actions
        self.Ai = np.ones(len(Si), len(A))  # discrete crisp actions for every state

        self.alpha = 0.8
        self.gamma = 0.75 # γ must be selected high enough so that the agent will try to collect long term rewards during an episode

        self.eligibility_trace = np.zeros(len(Si))
        self.lam = 0.5

        self.tau2t = 10

        pass

    # def get_Q_XtUt(self,q,aj,phi):
    #     """
    #     Gets the quality of continous action
    #     Qt(Xt,Ut) = sum{i=1;n}[qt(Si,aj)phi_i]
    #     :param q: full q table
    #     :param aj: selected action for each rule/fuzzy state
    #     :param phi:
    #     :return:
    #     """
    # def select_rule_actions(self,Xt):
    #     phi_i = FIS.make_inference(Xt) # FIS estimate rule strength
    #     q_eta = None
    #     q_rho = None
    #     q_total = self.q + q_eta + q_rho
    #     aj = np.argmax(q_total,axis=1)
    #     return aj,phi_i
    #
    # def get_continous_action(self,ai_prime,phi_i):
    #     """  Mixes action for each rule: Ut(Xt) = sum{i=1; n} ai_prime * phi_i
    #     :param ai_prime:
    #     :param phi_i:
    #     :return: Ut
    #     """
    #     Ut_Xt = np.sum(ai_prime*phi_i)
    #     return Ut_Xt
    #
    # def get_Utstar_Xt(self,phi_i, Xt):
    # #def get_optimal_continous_action(self,Xt):
    #     """Qstar(Xt) = sum{i=1;n}[max{aj\inA}[qt(Si,aj)*phi_i]]"""
    #     phi_i = FIS.make_inference(Xt)  # FIS estimate rule strength
    #     ai_star = np.argmax(self.q, axis = self.ax_action) # optimal action for each rule
    #     Ut_star = np.sum(self.Ai[self.Si, ai_star] * phi_i)
    #     return Ut_star
    #     # qi_astar = self.q[Si,ai_star] # optimal quality for each rule


    def eval_Ut(self,Ut):
        """Ut is given as [ai,phi_i] and this evalutes to continous global action"""
        ai, phi_i = Ut
        return np.sum(self.Ai[:,ai] * phi_i)


    def get_Qt_XtUt(self,Xt,Ut):
        """
        :param Xt:
        :param Ut: continous global action written as
        :return:
        """
        # phi_i = FIS.make_inference(Xt)  # FIS estimate rule strength
        phi_i = FIS.make_inference(Xt)  # FIS estimate rule strength
        if Ut == 'optimal':
            qi_astar = np.argmax(self.q, axis=self.ax_action)  # optimal quality for each rule
            Qtstar_Ut = np.sum(qi_astar * phi_i)
            return Qtstar_Ut
        else:
            ai, phi_i = Ut
            Qt_UtXt = np.sum(self.q[self.Si,ai] * phi_i)
            return Qt_UtXt


    def update_Q(self,et0,Xt0,Ut0,Xt1,rt1):
        """
        Updated each iteration of the algorithm
        :param et0: eligibility trace
        :param Xt0: state before new observation
        :param Ut0: action taken before new observation
        :param Xt1: new observation
        :param rt1: reward of new observation
        :return:
        """
        qt0 = self.q
        _, Q_XtUt0 = self.get_Qt_XtUt(Xt0, Ut = Ut0)
        _, Qstar_Xt1 = self.get_Qt_XtUt(Xt1, Ut = 'optimal')

        # for a rule is given by the Q * -function:
        td_error = rt1 + self.gamma * Qstar_Xt1 - Q_XtUt0
        qt1 = qt0 + self.alpha * td_error * et0

        self.q = qt1

    def update_EligibiltyTrace(self,Ut):
        ai,phi_i = Ut
        et = self.eligibility_trace
        et = (self.gamma*self.lam)*et # decay trace
        et[self.Si,ai] += phi_i       # add strength to current action eligibility
        self.eligibility_trace = et
        return self.eligibility_trace

    def get_reward(self,jerk_samples):
        """
        :param jerk_samples: tau2t jerk samples from faster admittance controller
        :param tau_f: duration of motion in discrete steps (unkown before hand)
        :param N: non-negative cumulative jerk over duration of discrete motion tau_f
        :return:
        """
        rt1 = np.power(np.linalg.norm(jerk_samples),2)
        N = np.abs(jerk_t)




def init_model():
    # Create FIS
    xi_tmp = []
    for key in FIS_settings:
        fset = FIS_settings[key]
        centers = np.linspace(fset['range'][0], fset['range'][1], fset['nbin'])
        size = (fset['range'][1] - fset['range'][0]) / (fset['nbin'] - 1)
        xtmp = StateVariable.InputStateVariable(
            *[FuzzySet.Triangles(cent - size, cent, cent + size) for cent in centers])
        xi_tmp.append(xtmp)

    # x1 = xi_tmp[0] x2 = xi_tmp[1]  x3 = xi_tmp[2] x4 = xi_tmp[3]
    fis = FIS.Build(*xi_tmp)

    model = FQL.Model(gamma=0.6,
                      alpha=0.05,
                      ee_rate=0.9,
                      q_initial_value='random',
                      action_set_length=3, fis=fis)
    return model

def main():
    """
    Ri: rule i,...,n
    A = {a1,..,aj}: action set containing j descrete numbers of crisp action values
    Xt: continous global state
    Ut: continous global action output composed of actions selected on each rule
    Si=>Ri: fuzzy state of the rule Ri composed by a vector if fuzzy sets
    phi_i: firing strength of rule Ri
    q(Si,aj): q-value that determines the probability of choosing the action
    ai_prime: selected actions for each rule given policy

    :return:
    """
    # test_human()

    # SET UP PLOTS
    plt.ion()
    nRows, nCols = 4,1
    lines = np.empty(nRows,nCols,dtype=object)
    names = np.empty(nRows, nCols, dtype=object)
    names[0, 0] = 'Epi Reward'
    names[1, 0] = 'Epi Length'
    names[2, 0] = 'V'
    names[3, 0] = 'D'
    fig,axs = plt.subplots(nRows,nCols)
    axs = np.array(axs).reshape(nRows,nCols)
    _x = np.arange(10)
    _y = np.ones(10)

    for r in range(nRows):
        for c in range(nCols):
            lines[r,c] = axs[r,c].plot(_x,_y)
            axs[r, c].set_ylable(names[r,c])

    # INTIALIZE TRAINING
    model = init_model()
    env = Environment(**env_params)
    time4update = lambda _itau: (_itau % FQL_settings['tau2t'] == 0)

    n_states = model.q_table.shape[0]
    n_actions = model.q_table.shape[1]
    qt = np.zeros(n_states,n_actions)
    et = np.zeros(n_states,n_actions)
    # eligibility = np.zeros(n_states)
    #et = EligibilityTrace(n_states,gamma=model.gamma,lam=0.95)


    stats = {}
    stats['epi_reward'] = []
    stats['epi_len'] = []
    stats['epi_velocities'] = []
    stats['epi_dampings'] = []


    for ith_episode in range(FQL_settings['num_episodes']):
        statei = env.reset()
        stats['epi_reward'].append(0)
        stats['epi_len'].append(0)
        stats['epi_velocities'] = []
        stats['epi_dampings'] = []
        statei = env.reset()


        for itau, tau in enumerate(FQL_settings['tau_iterable']):
            # -------- update policy --------
            if time4update(itau):




                if aj=aiprime:   et[Si,aj] = gamma* lam * et[Si,aj] + phi_i
                else: et[Si,aj] = gamma* lam * et[Si,aj]




                # global action U t given by the aggregation of all n rules
                Ut_Xt = np.sum(aiprime*phi_i)
                #A Q-function quantifies the quality of a given action with respect to the current state and is given by
                Q_XtUt = np.sum(qt[Si,aj]*phi_i)
                # The optimal  action
                Qstar_Xt = np.sum(np.max(qt[Si,:],axis=1)*phi_i)

                qnext = np.zeros(qt.shape)
                for i in range(n_states):
                    for j in range(n_actions):
                        qnext[i,j] = qt[i,j] + alpha* TD_err * et[i,j]

                qnext[Si,aj]



                # for a rule is given by the Q * -function:
                td_error = reward + gamma * Q[Si,aj] - state_values[state]
                state_values = state_values + alpha * td_error * eligibility
                #model.update_Qtbl(env.memory_dump())
                #statei = env.reset()
            # -----------------------------------------------------
            # update eligibility
            # for each action aj of rule Ri the trace et(Si,aj) is calculated as
            et[Si, :] = gamma * lam * et[Si, aj]  # decay all traces
            et[Si, aj] += phi_i



            actioni = model.get_action(statei)
            statej, reward, done, _ = env.step(actioni)

            # Close timestep
            env.memory_update() #memory.add(reward, v, dvdt, F, dFdt)
            stats['epi_reward'][-1] += reward
            stats['epi_len'][-1] = tau
            statei = np.copy(statej)

            if done: break

        # CLOSE EPISODE ===============================================


# Comu_reward=-1000000
# xtol = np.array([0.1,0.1,0.1])
# max_iter =int(2*env.N/10)
# nEpisodes = 50
#

stats = {}
stats['epi_reward'] = []
stats['epi_len'] = []





if __name__=="__main__":
    main()



