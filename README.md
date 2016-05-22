# Investigation into the use of neural networks to optimize the state space and Q-function performance of a Reinforcement Learning Problem
**Boris Repasky, Sina Eghbal, Suraj Narayanan S**

*{u5844485, u5544352, u5881495}@anu.edu.au*
   
Reinforcement Learning is generally used for problems where the system has a large number of states and has complex stochastic structure. 
A common approach for model-free reinforcement learning is Q-learning. At its simplest, Q-learning uses a look-up table to store data, 
which quickly looses viability with a very large number of state/action pairs. In this paper we aim to solve this problem by investigating 
the use of an auto-encoder neural network to compress the state space of a reinforcement learning problem. The non-linear features learned
by the autoencoder further enhances the preformance of vanilla Q-learning. Using the results we propose to do a comparative study to assess 
how effective auto-encoders are in preprocessing the state space. The study would consider several criteria such as number of non-linear
features used, learning rate, and overall performance to measure effectiveness. Future extensions to this research would be to 
have an on-line autoencoder to periodically update the state space by looking at states that have similar future expected rewards.

### How to run

Navigate to `code` folder, then use the following command:

`python pacman.py -p PacmanQAgent -x 2000 -n 2010 -a epsilon=0.05,gamma=0.8,alpha=0.2 -l mediumGrid -e autoenc -d state_file_mediumGrid_uniq.dat -u 10 -i 3000`

* `-u : number of hidden neurons`
* `-i : number of epochs to train the autoencoder`
* `-d : state space data file for training autoencoder`
* `-e : specify autoencoder module`

#### Layout, State space data mapping:

* `smallGrid     -> state_file_smallGrid_uniq.dat`
* `mediumGrid    -> state_file_mediumGrid_uniq.dat`
* `smallClassic  -> state_file_smallClassic_uniq.dat`
* `mediumClassic -> state_file_mediumClassic_uniq.dat`
