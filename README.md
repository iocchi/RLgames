# RLgames

Examples of use of RL agents for different games.

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
  -maxVfu MAXVFU    max visits for forward update of RA-Q tables [default: 0]
  -gamma GAMMA      discount factor [default: 1.0]
  -epsilon EPSILON  epsilon greedy factor [default: -1 = adaptive]
  -alpha ALPHA      alpha factor [default: -1 = based on visits]
  -niter NITER      stop after number of iterations [default: -1 = infinite]
  -rows ROWS        number of rows [default: 3]
  -cols COLS        number of columns [default: 3]
  --enableRA        enable Reward Automa
  --debug           debug flag
  --gui             GUI shown at start [default: hidden]
  --sound           Sound enabled
  --eval            Evaluate best policy
```

