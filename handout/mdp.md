## Markov Decision Processes (3 Exercises, 30 marks overall)

To get started, run ```Gridworld``` in manual control mode, which uses the arrow keys:

```
python gridworld.py -m
```

You will see the two-exit layout discussed in the slides.  The blue dot is the agent.  
Note that when you press ```up```, the agent only actually moves north 80% of the
time. Woe is him, such is the life of a ```Gridworld``` agent!

You can control many aspects of the simulation. A full list of options is available by running:

```
python gridworld.py -h
```

The default agent moves randomly

```
python gridworld.py -g MazeGrid
```

You should see the random agent bounce around the grid until it happens upon an exit.
Not the finest hour for an AI agent.

**Note:** The ```Gridworld``` _MDP_ is such that you first must enter a pre-terminal
state (the double boxes shown in the GUI) and then take the special 'exit' action before
the episode actually ends (in the true terminal state called ```TERMINAL_STATE```,
which is not shown in the GUI). If you run an episode manually, your total return
may be less than you expected, due to the discount rate (```-d``` to change; _0.9_ by default).

Look at the console output that accompanies the graphical output (or use ```-t``` for all text).
You will be told about each transition the agent experiences (to turn this off,
use ```-q```).

As in Pacman, positions are represented by _(x,y)_ Cartesian coordinates
and any arrays are indexed by ```[x][y]```, with ```'north'``` being
the direction of increasing ```y```, etc. By default,
most transitions will receive a reward of zero, though you can change this
with the living reward option (```-r```).

### Exercise 1 (15 marks)

Write a Value Iteration agent in the class ```ValueIterationAgent```, which has
been partially specified for you in the module
[```valueIterationAgents.py```](../code/valueIterationAgents.py). Your Value
Iteration agent is an off-line planner, not a Reinforcement Learning agent, and
so the relevant training option is the number of iterations of Value Iteration
it should run (option ```-i```) in its initial planning phase. ```ValueIterationAgent```
takes an instance of the class ```MDP``` on construction and runs Value Iteration
for the specified number of iterations before the constructor returns.

Value Iteration computes _k_-step estimates of the optimal values, V<sub>k</sub>. In
addition to running value iteration, implement the following methods
for ```ValueIterationAgent``` using V<sub>k</sub>.

 - ```getValue(s)``` returns the value of a state _s_, i.e. V<sub>k</sub>(_s_).
 - ```getPolicy(s)``` returns the **best** action _a_ according to the computed values V<sub>k</sub>.
 - ```getQValue(s, a)``` returns the Q-value of state and action pair _(s,a)_.

These quantities are all displayed in the GUI: values are numbers in squares, Q-values
are numbers in square quarters, and policies are arrows out from each square.

**Important:** Use the "batch" version of Value Iteration where each vector
V<sub>k</sub> is computed from a fixed vector V<sub>k-1</sub> (like in Lecture),
not the "online" version where one single weight vector is updated in place. The
difference is discussed in [Sutton & Barto book](http://www.cs.ualberta.ca/~sutton/book/ebook/node41.html)
in the 6th paragraph of chapter 4.1.

**Note:** A policy synthesized from values of depth _k_ (which reflect the next _k_ rewards)
will actually reflect the next _k+1_ rewards (i.e. you return
<img src="images/pi.png" alt="pi"/><sub>k+1</sub>).  Similarly, the Q-values
will also reflect one more reward than the values (i.e. you return Q<sub>k+1</sub>).
You may either return the synthesized policy <img src="images/pi.png" alt="pi"/><sub>k+1</sub>
or the actual policy for the kth iteration, <img src="images/pi.png" alt="pi"/><sub>k</sub>,
which you'll get if you store optimal actions from the most recent round of value iteration updates.

The following command loads your ```ValueIterationAgent```, computing a policy and executing
it 10 times

```
python gridworld.py -a value -i 100 -k 10
```

Press any key to cycle through values, Q-values, and the simulation.  
You should find that the value of the start state  (V(_start_), which you can read
off of the GUI) and the empirical resulting average reward (printed after the 10
rounds of execution finish) are quite close.

**Hint:** On the default ```BookGrid```, running value iteration for 5 iterations should give you this output:

```
python gridworld.py -a value -i 5
```

<center>
<img src="images/value.png" width="50%" alt="value iteration with k=5"/>
</center>

**Hint:** Use the ```util.Counter``` class in [```util.py```](../code/util.py),
which is a dictionary with a default value of zero.  Methods such as ```totalCount```
should simplify your code.  However, be careful with ```argMax```:
the actual argmax you want may be a key not in the counter!

**Grading:** Your Value Iteration agent will be graded on a new grid.  We will check
your values, Q-values, and policies after fixed numbers of iterations and at
convergence (e.g. after 100 iterations).

[Go back to Index of Handout](index.md)

### Exercise 2 (2.5 marks)

The layout  of ```BridgeGrid``` is a grid world map with the a low-reward terminal state and a
high-reward terminal state separated by a narrow "bridge", on either side of which
is a chasm of high negative reward. The agent starts near the low-reward state. With
the default discount of 0.9 and the default noise of 0.2, the optimal policy does not
cross the bridge. Change only ONE of the discount and noise parameters so that the
 optimal policy causes the agent to attempt to cross the bridge.  Put your answer
 in ```question2()```  of [```analysis.py```](../code/analysis.py).  
(_Noise_ refers to how often an agent ends up in an unintended successor state
  when they perform an action.)

The default corresponds to:

```
python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2
```

**Grading:** We will check that you only changed one of the given parameters, and
that with this change, a correct value iteration agent should cross the bridge.
Note that this part is tested with our Value Iteration agent, so if you have an
error in Value Iteration, your agent may cross the bridge for incorrect
settings of the parameters.

[Go back to Index of Handout](index.md)

### Exercise 3 (12.5 marks)

Consider the ```DiscountGrid``` layout, shown below. This grid has two
terminal states with positive payoff (shown in green), a close exit
with payoff +1 and a distant exit with payoff +10. The bottom row of
the grid consists of terminal states with negative payoff (shown in
red); each state in this "cliff" region has payoff -10. The starting
state is the yellow square. We distinguish between two types of
paths:

 1. paths that "risk the cliff" and travel near the bottom
  row of the grid; these paths are shorter but risk earning a large
  negative payoff, and are represented by the red arrow in the figure
  below.

 2. paths that "avoid the cliff" and travel along the top
  edge of the grid. These paths are longer but are less likely to
  incur huge negative payoffs. These paths are represented by the
  green arrow in the figure below.

<center>
<img src="images/discountgrid.png" width="50%" alt="DiscountGrid"/>
</center>

In this question, you will choose settings of the discount, noise, and living
reward parameters for this MDP to produce optimal policies of several different
types. Your setting of the parameter values for each part should have the property
that, if your agent followed its optimal policy without being subject to any
noise, it would exhibit the given behavior. If a particular behavior is not
achieved for any setting of the parameters, assert that the policy is
impossible by returning the string ```'NOT POSSIBLE'```.

Here are the optimal policy types you should attempt to produce:

 a. Prefer the close exit (+1), risking the cliff (-10)

 b. Prefer the close exit (+1), but avoiding the cliff (-10)

 c. Prefer the distant exit (+10), risking the cliff (-10)

 d. Prefer the distant exit (+10), avoiding the cliff (-10)

 e. Avoid both exits and the cliff (so an episode should never terminate)

You can check your answers by running Value Iteration on ```DiscountGrid``` with
specified discount, noise, and living reward parameters as follows:

```
python gridworld.py -a value -i 100 -g DiscountGrid --discount 0.9 --noise 0.2 --livingReward 0.0
```

and the answers given by implementing the functions ```question3a()``` through ```question3e()```
in [```analysis.py```](../code/analysis.py). Each of these should  return a 3-item tuple
of (discount, noise, living reward).

**Note:** You can check your policies in the GUI.  For example, using a correct
answer to 3(a), the arrow in _(0,1)_ should point east, the arrow in _(1,1)_ should
also point east, and the arrow in _(2,1)_ should point north.

**Grading:** We will check that the desired policy is returned in each case. As
in Exercise 2, we test your parameter settings using our value iteration agent.

[Go back to Index of Handout](index.md)
