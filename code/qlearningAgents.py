# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were originally developed at UC Berkeley,
# primarily by John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
#
# Further modifications and porting to Python 3 by Miquel Ramirez (miquel.ramirez@gmail.com),
# March and April 2016

""" Student Details
    Student Name: Suraj Narayanan S
    Student number: 5881495
    Date: 21/05/2015
"""

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

from time import clock
import random
import util

import numpy as np


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions
          for a state
    """

    def __init__(self, encoder=None, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Trained Autoencoder to extract non-linear features
        # from pacman state-space.
        if encoder:
            self.encoder = encoder['encoder']
            self.enc_type = encoder['enc_type']
        else:
            self.encoder = None

        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """

        # Encoding the original state to map to the relevant features
        # found by the autoencoder.
        if self.encoder:
            state = self.enc_type.predict(self.encoder, state)

        # Vanilla Q-learning
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0
        return self.q_values[(state, action)]

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # Vanilla Q-learning
        actions = self.getLegalActions(state)
        if not actions:
            return 0
        max_val = float("-inf")
        for action in actions:
            q_val = self.getQValue(state, action)
            if self.getQValue(state, action) > max_val:
                max_val = q_val
        return max_val

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        # Vanilla Q-learning
        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_actions = []
        for action in actions:
            if not max_actions:
                max_actions.append((action, self.getQValue(state, action)))
                continue
            if self.getQValue(state, action) > max_actions[0][1]:
                max_actions = [(action, self.getQValue(state, action))]
            elif self.getQValue(state, action) == max_actions[0][1]:
                max_actions.append((action, self.getQValue(state, action)))
        return random.choice(max_actions)[0]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        # Vanilla Q-learning
        actions = self.getLegalActions(state)
        if not actions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf

          Q(s,a) <- Q(s,a) + alpha(reward + gamma*Q(s',a') - Q(s,a))
        """

        # Vanilla Q-learning update algorithm
        q_val_cur = self.getQValue(state, action)
        q_val_nxt = self.getValue(nextState)
        update = self.alpha * (reward + self.discount * q_val_nxt - q_val_cur)

        # Apply update to the encoded state found by the autoencoder
        if self.encoder:
            state = self.enc_type.predict(self.encoder, state)

        # update
        self.q_values[(state, action)] += update


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class JarrydQAgent(QLearningAgent):

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.start_of_episode = True
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

        self.theta = util.Counter()
        self.visited_states = {}

    def phi(self, state, action):
        # Encoding the original state to map to the relevant features
        # found by the autoencoder.
        is_in_visited = (state, action) in self.visited_states
        if is_in_visited and 'phi' in self.visited_states[(state, action)]:
            return self.visited_states[(state, action)]['phi']
        else:
            phi_sa = self.enc_type.predict(self.encoder, state, action)
            if is_in_visited:
                self.visited_states[(state, action)]['phi'] = phi_sa
            else:
                self.visited_states[(state, action)] = {'phi': phi_sa}
            return phi_sa

    def Q(self, state, action):

        phi_sa = list(self.phi(state, action))
        theta = []
        for idx, f in enumerate(phi_sa):
            theta.append(self.theta[idx])
        theta = np.array(theta)
        return np.dot(theta, phi_sa)

    def maxQ(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # Vanilla Q-learning
        actions = self.getLegalActions(state)
        if not actions:
            return 0
        max_val = float("-inf")
        for action in actions:
            q_val = self.Q(state, action)
            if q_val > max_val:
                max_val = q_val
        return max_val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        #anneal alpha
        sa_alpha = self.alpha/self.visited_states[(state,action)]['pseudo_count']

        self.visited_states[(state, action)]['alpha'] = sa_alpha

        # start_time = clock()
        q_val_cur = self.Q(state, action)
        q_val_nxt = self.maxQ(nextState)
        td_err = reward + self.discount * q_val_nxt - q_val_cur
        update = self.alpha * td_err
        phi_sa = list(self.phi(state, action))
        for idx, f in enumerate(phi_sa):
            self.theta[idx] += update * f
        # elapsed = clock() - start_time
        # print("Time to calculate update:", elapsed)

    def d_phi_saCur_saVistied(self, sa_cur, sa_visited):
        # start_time = clock()
        euclidean_dist = np.array(self.phi(sa_cur[0], sa_cur[1])) - np.array(self.phi(sa_visited[0], sa_visited[1]))
        # elapsed = clock() - start_time
        # print("Euclidean Dist calculation:", elapsed)
        # start_time = clock()
        dist = np.dot(euclidean_dist, euclidean_dist)
        # elapsed = clock() - start_time
        # print("Dot Product:", elapsed)
        return dist

    def similarity_measure(self, sa_cur, sa_visited, gen_param=1):
        distance = self.d_phi_saCur_saVistied(sa_cur, sa_visited)
        return np.exp(-distance/(gen_param**2))

    def pseudo_count(self, state, action):
        sa_cur = (state, action)
        pseudo_count = 0
        # start_time = clock()
        for sa_visited, sa_obj in self.visited_states.items():
            if 'emperical_count' not in sa_obj:
                continue
            pseudo_count += self.similarity_measure(sa_cur, sa_visited) * sa_obj['emperical_count']
        # elapsed = clock() - start_time
        # print("Pseudo count calculation:", elapsed, len(self.visited_states))

        # if a state,action pair is visited increase its emperical count.
        if (state, action) in self.visited_states:
            self.visited_states[(state, action)]['pseudo_count'] = pseudo_count
        else:
            self.visited_states[(state, action)] = {'pseudo_count': pseudo_count}

        return pseudo_count

    def explore_bonus(self, state, action, bonus_scaling_factor=0.05):
        if self.start_of_episode:
            start_time = clock()
        exp_bonus = bonus_scaling_factor/(np.sqrt(self.pseudo_count(state, action)) + 0.01)
        if self.start_of_episode:
            elapsed = clock() - start_time
            print("Time to calculate explore_bonus:", elapsed)
            self.start_of_episode = False
        return exp_bonus

    def __value_for_action(self, state, action):
        return self.Q(state, action) + self.explore_bonus(state, action)

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_actions = []
        for action in actions:
            val_4_action = self.__value_for_action(state, action)
            if not max_actions:
                max_actions.append((action, val_4_action))
                continue
            if val_4_action > max_actions[0][1]:
                max_actions = [(action, val_4_action)]
            elif val_4_action == max_actions[0][1]:
                max_actions.append((action, val_4_action))
        return random.choice(max_actions)[0]

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        # start_time = clock()
        action = self.getPolicy(state)
        # elapsed = clock() - start_time
        # print("Time to calculate policy:", elapsed)
        self.doAction(state, action)

        #Update the emperical_count
        # if a state,action pair is visited increase its emperical count.
        is_in_visited = (state, action) in self.visited_states
        if is_in_visited and 'emperical_count' in self.visited_states[(state, action)]:
            self.visited_states[(state, action)]['emperical_count'] += 1
        else:
            if is_in_visited:
                self.visited_states[(state, action)]['emperical_count'] = 1
            else:
                self.visited_states[(state, action)] = {'emperical_count': 1}

        # Since the action has been taken increment the pseudo-count by 1
        if is_in_visited and 'pseudo_count' in self.visited_states[(state, action)]:
            self.visited_states[(state, action)]['pseudo_count'] += 1

        return action

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        self.start_of_episode = True
        if self.episodesSoFar%100==0:
                print('\n\n\n\nVisited State Object Dump\n', self.visited_states, '\n\n\n\n')
        print('Visited States Size:', len(self.visited_states))

        print('Score:', state.getScore())


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

        # Save states encountered when played by an intelligent agent.
        # These states are used by Autoencoder+Vanilla Q-learning.
        self.enable_sate_saving = False
        self.uniq_state = {}
        self.state_file = open('state_file_raw.dat', 'w')

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)

        # Start saving the states ones training has been completed.
        if self.enable_sate_saving:
            s = str(state).split('Score')[0]
            if s not in self.uniq_state:
                self.uniq_state[s] = 0
                self.state_file.write(s + action + '\n')

        self.doAction(state, action)
        return action

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        # Approximate Q-learning
        features = self.featExtractor.getFeatures(state, action)
        w, f = [], []
        for feature, value in features.items():
            w.append(self.weights[feature])
            f.append(value)
        w = np.array(w)
        f = np.array(f)
        return np.dot(w, f)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        # Approximate Q-learning
        q_val_cur = self.getQValue(state, action)
        q_val_nxt = self.getValue(nextState)
        correction = reward + self.discount * q_val_nxt - q_val_cur
        update = self.alpha * correction
        features = self.featExtractor.getFeatures(state, action)
        for feature, value in features.items():
            self.weights[feature] += update * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # Enable state saving
            print("Save states enabled.")
            self.enable_sate_saving = True
            pass
