## Approximate Q-learning and State Abstraction (2 Exercises, 20 marks overall)

## Exercise 8 (5 Marks)

Time to play some PacMan! Pacman will play games in two phases. In the first
phase, _training_, PacMan will begin to learn about the values of positions and actions.
Because it takes a very long time to learn accurate Q-values even for tiny grids,
PacMan's training games run in quiet mode by default, with no GUI (or console)
display.  Once Pacman's training is complete, he will enter _testing_ mode.  
When testing, Pacman's epsilon greedy policy parameter ```self.epsilon``` and
step-size ```self.alpha``` will be both set to 0.0, effectively stopping
Q-learning and disabling exploration, in order to allow PacMan to **exploit** his
learned policy. Test games are shown in the GUI by default.  Without any code
changes you should be able to run Q-learning Pacman for very tiny grids as follows:

```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

This is not a entirely trivial task. Compare the performance attained by your Q-learning
agent against what you can do right off the bat

```
python pacman.py -l smallGrid
```

you will appreciate that while we get the task "about right" with a few games,
Q-learning needs a quite huge number of trials before it starts playing _okay_.

Note that the class ```PacmanQAgent``` is already defined for you in terms of
the class ```QLearningAgent``` you've already written.  ```PacmanQAgent```
is only different in that it has default learning parameters that are more
effective for the Pacman problem (```epsilon=0.05, alpha=0.2, gamma=0.8```).  
You will receive full credit for this question if the command above works
without exceptions and your agent wins at least 80% of the time. The
automated tester will run 100 test games after the 2000 training games.

**Hint:** If your ```QLearningAgent``` works for [```gridworld.py```](../code/gridworld.py)
and [```crawler.py```](../code/crawler.py) but does not seem to be learning a
good policy for Pacman on ```smallGrid```, it may be because your ```getAction```
and/or ```getPolicy``` methods do not in some cases properly consider unseen
actions.  In particular, because unseen actions have by definition a Q-value of
zero, if all of the actions that **have** been seen have negative Q-values, an
unseen action may be optimal. Beware of the argmax function from ```util.Counter```!

**Note:** If you want to experiment with learning parameters, you can use the option
option ```-a```, for example

```
-a epsilon=0.1,alpha=0.3,gamma=0.7
```  

These values will then be accessible as ```self.epsilon```, ```self.gamma```
and ```self.alpha``` inside the agent.

**Note:** While a total of 2010 games will be played, the first 2000 games
will not be displayed because of the option <code>-x 2000</code>, which designates
the first 2000 games for training (no output).  Thus, you will only see Pacman
play the last 10 of these games.  The number of training games is also passed
to your agent as the option ```numTraining```.

**Note:** If you want to watch 10 training games to see what's going on, use the command:

```
python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10
```

During training, you will see output every 100 games with statistics about how
Pacman is faring. Epsilon is positive during training, so Pacman will play poorly
even after having learned a good policy: this is because he occasionally makes a
random exploratory move into a ghost. As a benchmark, it should take about 1,000
games  before Pacman's rewards for a 100 episode segment becomes positive, reflecting
that he's started winning more than losing. By the end of training, it should
remain positive and be fairly high (between 100 and 350).

Make sure you understand what is happening here: the MDP state is the **exact**
board configuration facing Pacman, with the now complex transitions describing
an entire ply of change to that state.  The intermediate game configurations in
which Pacman has moved but the ghosts have not replied are **not** _MDP_ states,
but are bundled in to the transitions. This technique is referred to as
_afterstates_ and is discussed in
[Sutton & Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node68.html).

Once Pacman is done training, he should win very reliably in test games
(at least 90% of the time), since now he is exploiting his learned policy.

However, you will find that training the same agent on the seemingly simple
[```mediumGrid```](../code/layouts/mediumGrid.lay)
does not work well. In our implementation, Pacman's average training rewards
remain negative throughout training.  At test time, he plays badly, probably
losing all of his test games.  Training will also take a long time, despite
its ineffectiveness.

Pacman fails to win on larger layouts because each board configuration is a
separate state with separate Q-values.  He has no way to generalize that
running into a ghost is bad for all positions.  Obviously, this approach
will not scale.

### Exercise 9 (15 marks)

Implement an approximate Q-learning agent that learns weights for features of
states, where many states might share the same features.  Write your implementation
in the class ```ApproximateQAgent``` in [```qlearningAgents.py```](../code/qlearningAgents.py),
which is a subclass of ```PacmanQAgent```.

The reasons why the exact PacmanQAgent fails for larger grids in the last problem
are complex to analyze and understand. One of the most intuitive observations that
can be made is that there are many irrelevant details in each state, and so the
same lessons must be learned many times.   However, just as with MiniMax, we can
compute (approximate) values based on aspects of the state.  For example, if a
ghost is about to eat Pacman, it's irrelevant which subset of the food in the
grid is present, so we should be able to learn much about ghost fear from the
ghost positions alone. Alternatively, if a ghost is sufficiently far away it
doesn't matter much exactly **how** far away or exactly where it is.  

**Note:**  Approximate Q-learning assumes the existence of a feature function
f(s,a) over state and action pairs, which yields a vector
f<sub>1</sub>(s,a) .. f<sub>i</sub>(s,a) .. f<sub>n</sub>(s,a) of feature values.
We provide feature functions for you in
[```featureExtractors.py```](../code/featureExtractors.py). Feature vectors are
instances of ```util.Counter```, dictionary-like objects containing the non-zero
pairs of features and values; all omitted features have value zero.

The approximate Q-function takes the following form

<center>
	<img  src="images/define-eqn1.png">
</center>

where each weight w<sub>i</sub> is associated with a particular feature f<sub>i</sub>(s,a).
In your code, you should implement the weight vector as a dictionary mapping
features (which the feature extractors will return) to weight values. You will
update your weight vectors similarly to how you updated Q-values:

<center>
	<br>
	<img  src="images/define-eqn2.png">
</center>

Note that the **correction** term is the same as in normal Q-learning.

By default, ```ApproximateQAgent``` uses class ```IdentityExtractor```,
whose instances assign a single feature to every _(state,action)_ pair.
With this feature extractor, your approximate Q-learning agent should work
identically to ```PacmanQAgent```.  You can test this with the following command:

```
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
```

**Important:** ```ApproximateQAgent``` is a subclass of  ```QLearningAgent```,
and it therefore shares several methods like ```getAction```.  Make sure that
your methods in ```QLearningAgent``` call ```getQValue``` instead of accessing
Q-values directly, so that when you override ```getQValue``` in your approximate
agent, the new approximate q-values are used to compute actions.

<Once you're confident that your approximate learner works correctly with the
identity features, run your approximate Q-learning agent with our custom
feature extractor, which can learn to win with ease:

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
```

Even much larger layouts should be no problem for your ```ApproximateQAgent```.
(**warning**: this may take a few minutes to train)

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
```

If you have no errors, your approximate Q-learning agent should win almost every
time with these simple features, even with only 50 training games.

**Grading:** We will run your agent on the ```mediumGrid``` layout 100 times using
a fixed random seed. You will receive 5 marks if your agent wins more than 25% of
its games, 10 marks if it wins more than 50% of its games, and 15 marks if
it wins more than 75% of its games. You can try your agent out under these conditions with

```
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 150 -l mediumGrid -q -f
```

_Congratulations!  You have a learning Pacman agent!_
