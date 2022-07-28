import numpy as np
from IPython.display import clear_output
import sys

from torch import ge
sys.path.append("..")

from collections import deque
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym import Env, spaces, utils
from gym.utils import seeding
### 

class gridWorld(Env):
    
    def __init__(self):
        
        self.side_length = [5,12]
        self.slippage = 0.
        self.f_perc = 0.6
        self.max_steps = 100
        env_size = [self.side_length[0], self.side_length[1]]
        self.env_size = env_size
        self.observable_size = self.env_size
        self.n_actions = 5


        ### GYM FEATURES
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.side_length[0], self.side_length[1], 1), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.reward_range = [0,1]
        ####

        self.size = env_size
        self.h = self.size[0]
        self.w = self.size[1]
        self.corner_idxs = [0,self.w-1,(self.h-1)*self.w,self.h*self.w-1]
        
        self.n_states = env_size[0]*env_size[1]+1
        self.action_dict = {"Up":0, "Right":1, "Down":2,"Left":3,"DO_NOTHING":4}
        self.create_dicts_and_indexes()
        self.random_state = np.random.RandomState(42)
        self.gym_view = np.array([
                        ['S', 'F', 'H', 'H', 'H', 'H', 'H', 'H','H','H','H','H'],
                        ['F', 'F', 'H', 'H', 'H', 'H', 'F', 'F','F','F','F','G'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'H','H','H','H','H'],
                        ['F', 'F', 'H', 'H', 'H', 'F', 'H', 'H','H','H','H','H'],
                        ['F', 'F', 'F', 'E', 'H', 'F', 'H', 'H','H','H','H','H']
                    ]
        )
        
        self.log_locations()
        self.starting_state = self.state
        self.create_board()
        self._init_probs_dict()
        self.n_steps = 0
    
    def log_locations(self):
        self.lakes = []
        
        for idx, i in enumerate(self.gym_view.flatten()):
            
            if i == 'H':
                self.lakes.append(idx)
            elif i == 'S':
                self.state = idx
            elif i == 'G':
                self.goal_state = idx
            elif i == 'E':
                self.erroneous_goal_state = idx

        self.lakes_coors = [self.stateIdx_to_coors[x] for x in self.lakes]
        self.state_coors = self.stateIdx_to_coors[self.state]
        self.goal_coors = self.stateIdx_to_coors[self.goal_state]
        self.erroneous_goal_coors = self.stateIdx_to_coors[self.erroneous_goal_state]
    
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state)

        return next_state, reward

        
    def reset(self , *, seed = None,  return_info: bool = False, options = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.state = self.starting_state
        self.state_coors = self.stateIdx_to_coors[self.state]
        self.create_board()
        self.n_steps = 0
        obs = self.board.copy()
        if np.random.uniform(0,1000) < 1:
            print(obs)
        return obs


    def step(self, action):
        try:
            if action < 0 or action >= self.n_actions:
                raise Exception('Invalid_action.')
        except:
            print("Here",action)
        self.n_steps += 1
        self.state, reward = self.draw(self.state, action)
        self.state_coors = self.stateIdx_to_coors[self.state]
        
        done = (self.n_steps >= self.max_steps) or (self.state == self.terminal_state)
        
        self.create_board()
        o = self.board.copy()
        return o, reward, done, {"st": [0,self.state]}


    def p(self,next_state, state, action):
        """
        Here, based on a 'chosen' action, we give the probability of transitioning from one state to another
        Functions:
          We calculate the probability if the chosen action is the 'actual' action, multiplied by chance of not taking random action
          We then add the probability for each action, multiplied by chance of taking random action / number of actions
        """
        no_rnd = 1 - self.slippage
        probas = 0
        probas += no_rnd * self.SAS_probs[state][action][next_state]
        for a in range(self.n_actions):
            probas += (self.slippage/self.n_actions) * self.SAS_probs[state][a][next_state]
        return probas
        "The method p returns the probability of transitioning from state to next state given action. "
        
    def r(self, next_state, state):
        "The method r returns the expected reward in having transitioned from state to next state given action."
        
        if state in [self.goal_state]:
            return 1
        elif state in [self.erroneous_goal_state]:
            return 0.5
        elif state in self.lakes:
            return 0
        else:
            return 0.0
    
    def render(self):
        print(self.board)
    
    def close(self):
        pass

    def seed(self, seed = None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def create_dicts_and_indexes(self):
        """
        Inputs... 
         size of lake (tuple e.g. (4,4))
         Location of lakes in coordinate form e.g. [(0,1),(1,2)...]
         Location of goal_states and their rewards e.g. {(3:3):1, (5,5):-1} In our examples this is always just one goal state

        Outputs...
         Dictionary linking coordinates to index of each state, and reverse dictionary
         Lake squares in index form e.g. [3,6,9]
         Goal states in index form e.g {15: 1, 25: -1}
        """
        
        self.coors_to_stateIdx = {}
        idx =0
        for r in range(self.h):
            for c in range(self.w):
                self.coors_to_stateIdx[(r,c)] = idx

                idx+=1

        self.coors_to_stateIdx[(-1,-1)] = self.n_states-1
        self.terminal_state = self.n_states-1

        self.stateIdx_to_coors = {}
        for k,v in self.coors_to_stateIdx.items():
            self.stateIdx_to_coors[v]=k
        


    def create_board(self):
        """
        Inputs: size of lake (h and w), coordinate location of lakes, and coordinate location and value of goal states
        Outputs: array of player-less board, with lake locations and reward locations
        """
        ### Creation of board object
        h,w = self.h,self.w
        self.board = np.array([0.0] * h*w).reshape(h,w)
        for l in self.lakes_coors:
            self.board[l] = -1.0
        
        self.board[self.goal_coors] = 1
        self.board[self.state_coors] = 0.5
        self.board = np.expand_dims(self.board,2)


    def _init_probs_dict(self):
        """
        In: the backend of the board (stateIdx_to_coors dict, lakes, goals, terminal state)
        Out: returns the impact of an ACTUAL action on the board position of a player
        Structure of output: {Current_State1: {Up: state1, state2, state 3....,
                            Down: state1, state2, state 3...}
                            ....
                    Current_State2: {Up ......}}
        
        note: 'actual' action distinguished here from 'chosen' action. Players 'choose', then we apply randomness, and then there is an 'actual' action
        This function concerns the effect of an 'actual' action on the position of a player.
        """
        
        ### HELPER FUNCTIONS
        def state_is_top(state):
            return self.stateIdx_to_coors[state][0] == 0
        def state_is_bottom(state):
            return self.stateIdx_to_coors[state][0] == self.h-1
        def state_is_left(state):
            return self.stateIdx_to_coors[state][1] == 0
        def state_is_right(state):
            return self.stateIdx_to_coors[state][1] == self.w-1
        def move_up():
            return -self.w
        def move_down():
            return self.w
        def move_left():
            return -1
        def move_right():
            return 1
        
        SA_prob_dict = {}
        lakes_and_goals = [self.goal_state] + [self.erroneous_goal_state] + self.lakes
        
        for state in range(self.n_states):
            SA_prob_dict[state] = {}
            #### Set the chance of entering an absorbing from lake or goal to 1
            for i in range(self.n_actions):
                SA_prob_dict[state][i] = np.zeros((self.n_states,))
                if state in lakes_and_goals or state == self.terminal_state:
                    SA_prob_dict[state][i][self.terminal_state] = 1
            
            if state not in lakes_and_goals and state != self.terminal_state:
                SA_prob_dict[state][self.action_dict['DO_NOTHING']][state] = 1
                """For UP"""
                if not state_is_top(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Up']][state+move_up()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Up']][state] = 1

                """For DOWN"""
                if not state_is_bottom(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Down']][state+move_down()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Down']][state] = 1

                """For LEFT"""
                if not state_is_left(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Left']][state+move_left()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Left']][state] = 1

                """For RIGHT"""
                if not state_is_right(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Right']][state+move_right()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Right']][state] = 1     
        self.SAS_probs = SA_prob_dict
        
    

if __name__ == '__main__':
    from config import Config as cfg
    c = cfg()
    
    for _ in range(2):
        env=gridWorld(c)
        done = False
        obs, state = env.reset()
        print(obs)
        while not done:
            
            act = int(input("give me an action"))
            obs, reward, done, state = env.step(act)
            print(obs)
            print(reward)
            print(done)
        

