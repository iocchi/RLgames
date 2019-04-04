# RLgames

Examples of use of RL agents for different games.

You can select a game, a RL algorithm and the training file. Learning mode will train the agent, evaluation mode (with option ``--eval``) will execute the current best policy.

### Requirements

The code requires Python 2.7.

To install the requirements:

    python2 -m pip install -r requirements.txt

### How to use

```
usage: game.py [-h] [-maxVfu MAXVFU] [-gamma GAMMA] [-epsilon EPSILON]
               [-alpha ALPHA] [-niter NITER] [-rows ROWS] [-cols COLS]
               [--enableRA] [--debug] [--gui] [--sound] [--eval]
               game agent trainfile

RL games

positional arguments:
  game              game (e.g., Breakout)
  agent             agent [Q, Sarsa, MC]
  trainfile         file for learning strctures

optional arguments:
  -h, --help        show this help message and exit
  -rows ROWS        number of rows [default: 3]
  -cols COLS        number of columns [default: 3]
  -gamma GAMMA      discount factor [default: 1.0]
  -epsilon EPSILON  epsilon greedy factor [default: -1 = adaptive]
  -alpha ALPHA      alpha factor (-1 = based on visits) [default: 0.5]
  -nstep NSTEP      n-steps updates [default: 0]
  -niter NITER      stop after number of iterations [default: -1 = infinite]
  --debug           debug flag
  --gui             GUI shown at start [default: hidden]
  --sound           Sound enabled
  --eval            Evaluate best policy
```

### Examples

Game: SimpleGrid 5x5
RL algorithm: Q-learning

```
python game.py SimpleGrid Q simplegrid55_Q_01 -rows 5 -cols 5 -gamma 0.9

```

Game: SimpleGrid 7x5
RL algorithm: Sarsa

```
python game.py SimpleGrid Sarsa simplegrid75_Sarsa_01 -rows 7 -cols 5 -gamma 0.9

```

Game: SimpleGrid 9x9
RL algorithm: Sarsa with n-steps updates

```
python game.py SimpleGrid Sarsa simplegrid99_Sarsa_n10_01 -rows 9 -cols 9 -gamma 0.9 -nstep 10

```



Game: BreakoutN (normal states) 3x3
RL algorithm: MC

```
python game.py BreakoutN MC breakoutN33_MC_01 -rows 3 -cols 3

```


Game: BreakoutN (normal states) 3x4
RL algorithm: Sarsa with n-step updates

```
python game.py BreakoutN Sarsa breakoutS34_S_n100_e01_01 -rows 3 -cols 4 -nstep 100 -epsilon 0.1

```


Game: BreakoutS (simplified states) 3x4
RL algorithm: MC

```
python game.py BreakoutS MC breakoutS34_MC_01 -rows 3 -cols 4

```




You can stop the process at any time are resume it later.
If you want to execute another experiment, just change the training filename.


To evaluate the current best policy, use the same command line adding ```--gui --eval```.

Example:

```
python game.py BreakoutS MC breakoutS34_MC_01 -rows 3 -cols 4 --gui --eval

```

### Plotting the results

```
python plotresults.py <trainfile>

```

Example:

```
python plotresults.py breakoutS34_MC_01

```


