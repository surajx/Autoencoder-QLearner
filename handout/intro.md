## Introduction

In this project, you will implement Value Iteration and Q-learning. You will test your agents first on ```Gridworld``` (from class), then apply them to a simulated robot controller (```Crawler```) and ```Pacman```.

The code for this project contains the following files, which are available in the
folder ```code``` of this repository.

**Note:** Througout the handout, we will refer to the Python interpreter executable
as ```python```. Change this to match the setup on the machine you're working. For instance,
in ```Ubuntu 14.04``` you'll need to use ```python3``` instead of ```python``` since
the latter is pointing to the Python 2.x interpreter.

### Files you will edit and you NEED to submit

- [```valueIterationAgents.py```](../code/valueIterationAgents.py) - A
Value Iteration agent for solving known Markov Decision Processes (*MDPs*)

- [```qlearningAgents.py```](../code/qlearningAgents.py) - Q-Learning agents
for ```GridWorld```, ```Crawler``` and ```PacMan```.

- [```analysis.py```](../code/analysis.py) - A file to put your answers to
questions given in the handout.

### Files you SHOULD read but NOT edit

- [```mdp.py```](../code/mdp.py) - Defines methods on general *MDPs*.

- [```learningAgents.py```](../code/learningAgents.py) - Defines the base
classes ```ValueEstimationAgent``` and ```QLearningAgent```, which your agents
will extend.

- [```util.py```](../code/util.py) - Utilities, including ```util.Counter```,
which is particularly useful for Q-learners.

- [```gridworld.py```](../code/gridworld.py) - The ```GridWorld``` implementation.

- [```pacman.py```](../code/pacman.py) - The implementation of the ```PacMan```
game.

- [```featureExtractors.py```](../code/featureExtractors.py) - Classes for
extracting features on _(state,action)_ pairs. Used for the approximate Q-Learning
agent (in [```qlearningAgents.py```](../code/qlearningAgents.py)).

- [```game.py```](../code/game.py) - Classes used to represent Agents and layouts.

### Files you CAN ignore

- [```environment.py```](../code/environment.py) - Abstract class for general
reinforcement learning environments. Used by [```gridworld.py```](../code/gridworld.py).

- [```graphicsGridworldDisplay.py```](../code/graphicsGridworldDisplay.py) -  ```Gridworld``` graphical display.

- [```graphicsUtils.py```](../code/graphicsUtils.py) - Assorted graphics
utilities.

- [```textGridworldDisplay.py```](../code/textGridworldDisplay.py) - Plug-in
for the ```Gridworld``` text interface.

- [```crawler.py```](../code/crawler.py) - The ```Crawler``` code and test
harness. You will run this but not edit it.

- [```graphicsCrawlerDisplay.py```](../code/graphicsCrawlerDisplay.py) -
Graphical display of the crawler robot.

- [```ghostAgents.py```](../code/ghostAgents.py) - Hand-coded policies to control the
ghosts in PacMan.

- [```pacmanAgents.py```](../code/pacmanAgents.py) - Hand-coded policies for the PacMan
agent, not used in the assignment (useful for testing).

- [```keyboardAgents.py```](../code/keyboardAgents.py) - Classes implementing "live"
controllers (i.e. the control is via commands entered with the keyboard).

### What to Submit

You will fill in portions of the following files:

 - [```valueIterationAgents.py```](../code/valueIterationAgents.py)
 - [```qlearningAgents.py```](../code/qlearningAgents.py)
 - [```analysis.py```](../code/analysis.py)

during the assignment. You should submit **only** these files.  Please don't change any others.</p>

### Evaluation

Your code will be run against an automated testing suite to determine its technical
correctness. Please <em>do not</em> change the names of any provided functions or classes
within the code, or you will wreak havoc on the automated tester. If your code works
correctly on one or two of the provided examples but doesn't get full credit from the auto
tester, you most likely have a subtle bug that breaks one of our more thorough test cases;
you will need to debug more fully by reasoning about your code and trying small examples of
your own. That said, bugs in the auto tester are not impossible, so please do contact the
relevant staff if you believe that there has been an error in the grading.


### Getting Help

If you find yourself stuck on something, contact the lab tutors (David Nalder and
Leon Sheldon) for help. Wattle forums are there for your support; please use them.
We want these projects to be rewarding and instructional, not frustrating and
demoralising.  But, we don't know when or how to help unless you ask.
