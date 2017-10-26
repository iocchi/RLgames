#!/usr/bin/python

import importlib
import pygame, sys
import numpy as np
import atexit
import math, random, time
from math import fabs
import argparse

trainfilename = 'default'

args = None
game = None
agent = None


def loadGameModule():
    print("Loading game "+args.game)
    try:
        mod = importlib.import_module(args.game)
        if (args.game=='Breakout'):
            game = mod.Breakout(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename, enableRA = args.enableRA)
        if (args.game=='SimpleGrid'):
            game = mod.SimpleGrid(args.rows, args.cols, trainfilename)
    except:
        print "ERROR: game ",args.game," not found."
        raise
        sys.exit(1)
    return game


def loadAgentModule():
    print("Loading agent "+args.agent)
    try:
        modname = args.agent+'Agent'
        mod = importlib.import_module(modname)
        agent = mod.Agent()
    except:
        print "ERROR: agent ",modname," not found."
        raise
        sys.exit(1)
    return agent

        
@atexit.register
def save():
    if game is not None and agent is not None and (not args.eval):
        # filename = trainfilename +"_%05d" % (self.iteration)
        filename = 'data/'+trainfilename
        savedata = [int(game.iteration), int(game.hiscore), int(game.hireward), agent.savedata()]
        np.savez(filename, iter = savedata[0], hiscore = savedata[1], hireward = savedata[2], agentdata = savedata[3])
        print "Data saved successfully on file ", filename, "\n\n\n"

        
def load(fname, game, agent):
    data = None
    try:
        fn = 'data/'+str(fname)+'.npz'
        data = np.load(fn)
        s = "Data loaded from " + fn + " successfully."
        print(s)
    except IOError:
        s = "Error: can't find file or read data from file " + fn +" -> initializing a new Q matrix"
        print(s)

    if data is not None:
        try:
            game.iteration = int(data['iter'])
            game.hiscore = int(data['hiscore'])
            game.hireward = int(data['hireward'])
            agent.loaddata(data['agentdata'])
        except:
            print("Can't load data from input file, wrong format.")
            #raise

def execution_step(game, agent):
    x = game.getstate() # current state
    if (game.isAuto):  # agent choice
        a = agent.decision(x) # current action
    else: # otherwise command is set by user input
        a = game.getUserAction() # action selected by user
    game.update(a)
    x2 = game.getstate() # new state
    r = game.getreward() # reward
    agent.notify(x,a,r,x2)


# learning process
def learn(game, agent):

    run = True    
    while (run and (args.niter<0 or game.iteration<=args.niter) and not game.userquit):

        game.reset() # increment game.iteration
        game.draw()
        time.sleep(game.sleeptime)
        while (run and not game.finished):
            run = game.input()
            if game.pause:
                time.sleep(1)
                continue
            execution_step(game, agent)
            game.draw()
            time.sleep(game.sleeptime)

        # episode finished
        if (game.finished): 
            agent.notify_endofepisode(game.iteration)
            game.print_report()
            time.sleep(game.sleeptime)

        # end of experiment
        if (agent.optimal and game.goal_reached()):
            run = False

    game.quit()


# evaluation process
def evaluate(game, agent, n): # evaluate best policy n times (no updates)
    i=0
    run = True
    game.sleeptime = 0.05
    while (i<n and run):
        game.reset()
        game.draw()
        time.sleep(game.sleeptime)

        agent.optimal = True
        while (run and not game.finished):
            run = game.input()
            execution_step(game, agent)
            game.draw()
            time.sleep(game.sleeptime)
        game.print_report(printall=True)
        i += 1
    agent.optimal = False
    
# main

parser = argparse.ArgumentParser(description='RL for breakout game')
parser.add_argument('game', type=str, help='game (e.g., Breakout)')
parser.add_argument('agent', type=str, help='agent (e.g., RL for RLAgent.py)')
parser.add_argument('trainfile', type=str, help='file for learning strctures')
parser.add_argument('-maxVfu', type=int, help='max visits for forward update of RA-Q tables [default: 0]', default=0)
parser.add_argument('-gamma', type=float, help='discount factor [default: 1.0]', default=1.0)
parser.add_argument('-epsilon', type=float, help='epsilon greedy factor [default: 0.1]', default=0.1)
parser.add_argument('-alpha', type=float, help='alpha factor [default: -1 = based on visits]', default=-1)
parser.add_argument('-niter', type=float, help='stop after number of iterations [default: -1 = infinite]', default=-1)
parser.add_argument('-rows', type=int, help='number of rows [default: 3]', default=3)
parser.add_argument('-cols', type=int, help='number of columns [default: 3]', default=3)
parser.add_argument('--enableRA', help='enable Reward Automa', action='store_true')
parser.add_argument('--debugQ', help='debug Q updates', action='store_true')
parser.add_argument('--debug', help='debug flag', action='store_true')
parser.add_argument('--gui', help='GUI shown at start [default: hidden]', action='store_true')
parser.add_argument('--sound', help='Sound enabled', action='store_true')
parser.add_argument('--eval', help='Evaluate best policy', action='store_true')

args = parser.parse_args()

trainfilename = args.trainfile.replace('.npz','')

# load game and agent modules
game = loadGameModule()
agent = loadAgentModule()

# set parameters
game.debug = args.debug
game.gui_visible = args.gui
game.sound_enabled = args.sound

agent.gamma = args.gamma
agent.epsilon = args.epsilon
agent.alpha = args.alpha
agent.maxVfu = args.maxVfu
agent.debug = args.debug

game.init(agent)

# load saved data
load(trainfilename,game,agent)
print "Game iteration: ",game.iteration

# learning or evaluation process
if (args.eval):
    evaluate(game, agent, 10)
else:    
    learn(game, agent)

print "Experiment terminated !!!"
print "\n"
