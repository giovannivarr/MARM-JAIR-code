from reward_machines.reward_machine_utils import evaluate_dnf, value_iteration
from reward_machines.reward_functions import ConstantRewardFunction
import os

class RewardMachine:
    def __init__(self, file):
        # <U,u0,delta_u,delta_r>
        self.U = []  # list of non-terminal RM states
        self.u0 = None  # initial state
        self.delta_u = {}  # state-transition function
        self.delta_r = {}  # reward-transition function
        # self.terminal_u = -1  # All terminal states are sent to the same terminal state with id *-1*
        
        self.fail_state = -2    # fail state for when the agent falls out of the RM
        self._add_state([self.fail_state])
        self.delta_u[self.fail_state] = {}
        self.delta_r[self.fail_state] = {}
        self.delta_u[self.fail_state][self.fail_state] = 'True'
        self.delta_r[self.fail_state][self.fail_state] = ConstantRewardFunction(-1) # the fail state keeps the RM looping in it and gives a reward of -1
        
        self.terminal_states = []  # list of terminal RM states
        self._load_reward_machine(file)
        self.known_transitions = {}  # Auxiliary variable to speed up computation of the next RM state

        self.current_state = self.u0  # current state of the RM
        self.state_features = {}  # RM states represented as one-hot encodings
        for i, u in enumerate(self.U):
            self.state_features[u] = [0] * len(self.U)
            self.state_features[u][i] = 1

    # Public methods -----------------------------------

    def add_reward_shaping(self, gamma, rs_gamma):
        """
        It computes the potential values for shaping the reward function:
            - gamma(float):    this is the gamma from the environment
            - rs_gamma(float): this gamma that is used in the value iteration that compute the shaping potentials
        """
        self.gamma = gamma
        self.potentials = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_states, rs_gamma)
        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]

    def reset(self):
        # Reinitialize the RM to the initial state
        self.current_state = self.u0
        return self.u0

    def _compute_next_state(self, u1, true_props):
        for u2 in self.delta_u[u1]:
            if evaluate_dnf(self.delta_u[u1][u2], true_props):
                return u2
        return u1   # no transition is defined for true_props, so we remain in the same state
        #print('fell out!')
        #return self.fail_state  # no transition is defined for true_props, so go to fail state

    def get_next_state(self, u1, true_props):
        # Note: we can't allow a transition to a terminal state if there is no transition given
        # the current true propositions, as it might be that agents get a reward even though they shouldn't.
        # In order to avoid this, I've changed the return in the _compute_next_state method to return u1 instead of
        # the terminal state if no transition is defined given the current true propositions.
        if (u1, true_props) not in self.known_transitions:
            u2 = self._compute_next_state(u1, true_props)
            self.known_transitions[(u1, true_props)] = u2
        return self.known_transitions[(u1, true_props)]

    def step(self, u1, true_props, s_info, add_rs=False, env_done=False):
        """
        Emulates a step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        # This assertion is stupid, if the RM is in the terminal state it should loop there in our case
        # assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        if u1 in self.terminal_states:
            u2 = u1
            done = True
            rew = 0
        else:
            u2 = self.get_next_state(u1, true_props)
            done = (u2 in self.terminal_states)
            # Getting the reward
            rew = self._get_reward(u1, u2, s_info, add_rs)

        # Updating the current state
        self.current_state = u2

        return u2, rew, done

    def get_states(self):
        return self.U

    def get_current_state_feature(self):
        return self.state_features[self.current_state]

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]

    # Private methods -----------------------------------

    def _get_reward(self, u1, u2, s_info, add_rs):
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0 if u2 != -2 else -1 # NOTE: if the agent falls from the reward machine it receives a punishment of -1
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward += self.delta_r[u1][u2].get_reward(s_info)
        # Adding the reward shaping (if needed)
        rs = 0.0
        if add_rs:
            # un = self.terminal_u if env_done else u2  # If the env reached a terminal state, we have to use the potential from the terminal RM state to keep RS optimality guarantees
            rs = self.gamma * self.potentials[u2] - self.potentials[u1]
        # Returning final reward
        return reward + rs

    def _load_reward_machine(self, file):
        """
        Example:
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        self.terminal_states = eval(lines[1])
        #self.terminal_states.append(self.fail_state)

        # adding transitions
        for e in lines[2:]:
            # Reading the transition
            u1, u2, dnf_formula, reward_function = eval(e)
            # terminal states
            #if u1 in terminal_states:
            #    continue
            #if u2 in terminal_states:
            #    u2 = self.terminal_u
            # Adding machine state
            self._add_state([u1, u2])
            # Adding state-transition to delta_u
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
            if u2 not in self.delta_u:
                self.delta_u[u2] = {}
            self.delta_u[u1][u2] = dnf_formula
            # Adding reward-transition to delta_r
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
            self.delta_r[u1][u2] = reward_function
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U:
                self.U.append(u)
