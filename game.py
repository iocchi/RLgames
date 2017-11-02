#!/usr/bin/python

import importlib
import pygame, sys
import numpy as np
import atexit
import math, random, time
from math import fabs
import argparse

trainfilename = 'default'
optimalPolicyFound = False 

args = None
game = None
agent = None



def loadGameModule():
    print("Loading game "+args.game)
    try:
        if (args.game=='SimpleGrid'):
            mod = importlib.import_module(args.game)
            game = mod.SimpleGrid(args.rows, args.cols, trainfilename)
        elif (args.game=='BreakoutS'):
            mod = importlib.import_module('Breakout')
            game = mod.BreakoutS(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutN'):
            mod = importlib.import_module('Breakout')
            game = mod.BreakoutN(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutSRA'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutSRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutNRA'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
    except:
        print "ERROR: game ",args.game," not found."
        raise
        sys.exit(1)
    return game


def loadAgentModule():
    print("Loading agent "+args.agent)
    try:
        if (args.agent=='Q'):
            modname = 'RLAgent'
            mod = importlib.import_module(modname)
            agent = mod.QAgent()
        elif (args.agent=='Sarsa'):
            modname = 'RLAgent'
            mod = importlib.import_module(modname)
            agent = mod.SarsaAgent()
        if (args.agent=='MC'):
            modname = 'RLMCAgent'
            mod = importlib.import_module(modname)
            agent = mod.MCAgent()
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
        s = "Error: can't find file or read data from file " + fn +" -> initializing new structures"
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

            
def writeinfo(trainfilename,game,agent,init=True):
    global optimalPolicyFound
    infofile = open("data/"+trainfilename +".info","a+")
    allinfofile = open("data/all.info","a+")
    if (init):
        infofile.write("Train:   %s\n" %(trainfilename))
        infofile.write("Game:    %s\n" %(args.game))
        infofile.write("Size:    %d x %d\n" %(args.rows, args.cols))
        infofile.write("Agent:   %s\n" %(agent.name))
        infofile.write("gamma:   %f\n" %(agent.gamma))
        infofile.write("epsilon: %f\n" %(agent.epsilon))
        infofile.write("alpha:   %f\n" %(agent.alpha))
        infofile.write("n-step:  %d\n" %(agent.nstepsupdates))
        infofile.write("lambda:  %f\n\n" %(agent.lambdae))
        infofile.write("\n%s,%s,%d,%d,%s,%f,%f,%f,%d,%f\n" %(trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae))
        allinfofile.write("%s,%s,%d,%d,%s,%f,%f,%f,%d,%f\n" %(trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae))
    elif optimalPolicyFound:
        infofile.write("Optimal policy found.\n")
        infofile.write("goal reached:     %d\n" %(game.iteration))
        infofile.write("goal score:       %d\n" %(game.score))
        infofile.write("goal reward:      %d\n" %(game.cumreward))
        infofile.write("goal n. actions:  %d\n\n" %(game.numactions))
        infofile.write("\n%s,%s,%d,%d,%s,%f,%f,%f,%d,%f,%d,%d,%d,%d\n" %(trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae,game.iteration,game.score,game.cumreward,game.numactions))
        allinfofile.write("%s,%s,%d,%d,%s,%f,%f,%f,%d,%f,%d,%d,%d,%d\n" %(trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae,game.iteration,game.score,game.cumreward,game.numactions))
    else:
        infofile.write("last iteration:   %d\n" %(game.iteration))
        infofile.write("last score:       %d\n" %(game.score))
        infofile.write("last reward:      %d\n" %(game.cumreward))
        infofile.write("last n. actions:  %d\n\n" %(game.numactions))
        infofile.write("\n%s,%s,%d,%d,%s,%f,%f,%f,%d,%f,,,,,%d,%d,%d,%d\n" %(trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae, game.iteration,game.score,game.cumreward,game.numactions))
        allinfofile.write("%s,%s,%d,%d,%s,%f,%f,%f,%d,%f,,,,,%d,%d,%d,%d\n" %(trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae, game.iteration,game.score,game.cumreward,game.numactions))
    
    infofile.flush()
    infofile.close()
    allinfofile.flush()
    allinfofile.close()

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
    global optimalPolicyFound
    
    run = True
    last_goalreached = False
    next_optimal = False
    if (game.iteration>0): # try an optimal run
        next_optimal = True
        
    while (run and (args.niter<0 or game.iteration<=args.niter) and not game.userquit):

        game.reset() # increment game.iteration
        game.draw()
        time.sleep(game.sleeptime)
        if (last_goalreached or next_optimal):
            agent.optimal = True
            next_optimal = False
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
            optimalPolicyFound = True

        last_goalreached = game.goal_reached()

    if optimalPolicyFound:
        print("\n****************************")
        print("*** Optimal policy found ***")
        print("****************************\n")


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

parser = argparse.ArgumentParser(description='RL games')
parser.add_argument('game', type=str, help='game (e.g., Breakout)')
parser.add_argument('agent', type=str, help='agent [Q, Sarsa, MC]')
parser.add_argument('trainfile', type=str, help='file for learning strctures')
parser.add_argument('-rows', type=int, help='number of rows [default: 3]', default=3)
parser.add_argument('-cols', type=int, help='number of columns [default: 3]', default=3)
parser.add_argument('-gamma', type=float, help='discount factor [default: 1.0]', default=1.0)
parser.add_argument('-epsilon', type=float, help='epsilon greedy factor [default: -1 = adaptive]', default=-1)
parser.add_argument('-alpha', type=float, help='alpha factor (-1 = based on visits) [default: -1]', default=-1)
parser.add_argument('-nstep', type=int, help='n-steps updates [default: 0]', default=0.5)
parser.add_argument('-lambdae', type=float, help='lambda eligibility factor [default: -1 (no eligibility)]', default=-1)
parser.add_argument('-niter', type=float, help='stop after number of iterations [default: -1 = infinite]', default=-1)
parser.add_argument('--debug', help='debug flag', action='store_true')
parser.add_argument('--gui', help='GUI shown at start [default: hidden]', action='store_true')
parser.add_argument('--sound', help='Sound enabled', action='store_true')
parser.add_argument('--eval', help='Evaluate best policy', action='store_true')
#parser.add_argument('--enableRA', help='enable Reward Automa', action='store_true')
#parser.add_argument('-maxVfu', type=int, help='max visits for forward update of RA-Q tables [default: 0]', default=0)

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
agent.nstepsupdates = args.nstep
agent.lambdae = args.lambdae
#agent.maxVfu = args.maxVfu
agent.debug = args.debug

game.init(agent)

# load saved data
load(trainfilename,game,agent)
print "Game iteration: ",game.iteration

if (game.iteration==0):
    writeinfo(trainfilename,game,agent,init=True)

# learning or evaluation process
if (args.eval):
    evaluate(game, agent, 10)
else:    
    learn(game, agent)
    writeinfo(trainfilename,game,agent,init=False)

print "Experiment terminated !!!\n"

game.quit()
