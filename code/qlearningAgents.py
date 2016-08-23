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
        # self.enable_sate_saving = False
        # self.uniq_state = {}
        # self.state_file = open('state_file_raw.dat', 'w')

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        # Start saving the states ones training has been completed.
        # if self.enable_sate_saving:
        #     s = str(state)
        #     if s not in self.uniq_state:
        #         s = s.split('Score')[0]
        #         self.uniq_state[s] = 0
        #         self.state_file.write(s + '\n')

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
            # print("Save states enabled.")
            # self.enable_sate_saving = False
            pass
