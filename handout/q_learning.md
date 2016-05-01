## Q-learning (4 Exercises, 50 marks overall)

Note that your Value Iteration agent does not actually learn from experience. Rather,
it ponders its _MDP_ model to arrive at a complete policy before ever interacting
with a real environment. When it does interact with the environment, it simply
follows the precomputed policy (e.g. it becomes a reflex agent).  This distinction
may be subtle in a simulated environment like a ```GridWorld```, but it's very
important in the real world, where the real or full description of the
_MDP_ may not be readily available.

### Exercise 4 (30 Marks)

You will now write a Q-learning agent, which does very little on construction, but
instead learns by trial and error from interactions with the environment
through its ```update(state, action, nextState, reward)``` method. A stub of a
Q-learner is specified in the class ```QLearningAgent``` you can find in
the module [```qlearningAgents.py```](../code/qlearningAgents.py) ,
and you can select it with the option ```'-a q'```.  For this Exercise, you must
implement the methods

 - ```update(state, action, nextState, reward)```
 - ```getValue(state)```,
 - ```getQValue(state, action)```
 - and ```getPolicy(state)```.

**Note:** For ```getPolicy```, you should break ties randomly for better behavior.
The standard Python ```random.choice()``` function will help.  In a particular state,
actions that your agent **hasn't** seen before still have a Q-value, specifically
a Q-value of zero, and if all of the actions that your agent **has** seen before
have a negative Q-value, an unseen action may be optimal.

**Important:** Make sure that in your ```getValue``` and ```getPolicy``` functions,
you only access Q values by calling ```getQValue```. This
abstraction will be useful for Exercise 9 when you
override ```getQValue``` to use features of state-action
pairs rather than state-action pairs directly.

With the Q-learning update in place, you can watch your Q-learner learn under
manual control, using the keyboard:

```
python gridworld.py -a q -k 5 -m
```

Recall that ```-k``` will control the number of episodes your agent gets to learn.
Watch how the agent learns about the state it was just in, not the one it moves
to, and "leaves learning in its wake."

**Hint:** to help with debugging, you can turn off noise by using the ```--noise 0.0```
parameter (though this obviously makes Q-learning less interesting). If you manually
steer PacMan north and then east along the optimal path for four episodes, you should
see the following Q-values:

<center>
<img src="images/q-learning.png" width="50%" alt="QLearning"/>
</center>

**Note:** Q-learning may look to you _shockingly_ slow when it comes to converge to
quasi-optimal Q-values, even when one is steering the learning *manually*
towards the optimal policy. Take into account that you, by just looking at
the grid, can figure out quite quickly what needs to be done. The Q-learning
agent has to *discover* the right answer from first principles.

**Grading:** We will run your Q-learning agent on an example of our own and
check that it learns the same Q-values and policy as our reference implementation
when each is presented with the same set of examples.

[Go back to Index of Handout](index.md)

### Exercise 5 (10 marks)

Complete your Q-learning agent by implementing epsilon-greedy action selection
in ```getAction```, meaning it chooses random actions an epsilon fraction of the
time, and follows its current best Q-values otherwise.

```
python gridworld.py -a q -k 100
```

You may want to speed up the proceedings by using the option ```-s 1.0```.

Your final Q-values should resemble those of your Value Iteration agent, especially
along well-traveled paths.  However, your average returns will be lower than the
Q-values predict because of the random actions and the initial learning phase.

You can choose an element from a list uniformly at random by calling
the ```random.choice``` function. You can simulate a binary variable with probability _p_
of success by using ```util.flipCoin(p)```, which returns ```True``` with
probability _p_ and ```False``` with probability _1-p_.

### Exercise 6 (5 marks)

First, train a completely random Q-learner with the default learning
rate on the noiseless BridgeGrid for 50 episodes and observe whether it finds
the optimal policy.


```
python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1
```

Now try the same experiment with an epsilon of 0. Is there an epsilon and a
learning rate for which it is highly likely (greater than 99%) that the optimal
policy will be learned after 50 iterations? ```question6()``` in
[```analysis.py```](../code/analysis.py) should return **EITHER** a 2-item tuple of

```
(epsilon, learning rate)
```

**OR** the string

```
'NOT POSSIBLE'
```

if there is none.  Epsilon is controlled by ```-e```, learning rate by ```-l```.

**Note:** Your response should not depend on the exact
tie-breaking mechanism used to choose actions. This means your
answer should be correct even if for instance we rotated the entire
bridge grid world 90 degrees.

[Go back to Index of Handout](index.md)

### Exercise 7 (5 marks)

With no additional code, you should now be able to run a Q-learning crawler robot:

```
python crawler.py
```

If this doesn't work, you've probably written some code too specific to
the ```GridWorld``` problem and you should make it more general to all _MDPs_.

This will invoke the crawling robot from class using your Q-learner. Play around
with the various learning parameters to see how they affect the agent's policies
and actions.  Note that the step delay is a parameter of the simulation, whereas
the learning rate and epsilon are parameters of your learning algorithm, and
the discount factor is a property of the environment.

**Grading:** We give you 5 marks for free here, but play around with the crawler anyway!

[Go back to Index of Handout](index.md)
