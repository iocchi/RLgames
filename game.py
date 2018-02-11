#!/usr/bin/python

import importlib
import pygame, sys
import numpy as np
import atexit
import math, random, time
from math import fabs
from time import gmtime, strftime
import argparse

trainfilename = 'default'
optimalPolicyFound = False 

args = None
game = None
agent = None



def loadGameModule():
    print("Loading game %s" %args.game)
    try:
        if (args.game=='SimpleGrid'):
            mod = importlib.import_module(args.game)
            game = mod.SimpleGrid(args.rows, args.cols, trainfilename)
        elif (args.game=='BreakoutS'):
            mod = importlib.import_module('Breakout')
            game = mod.BreakoutS(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutFS'):
            mod = importlib.import_module('Breakout')
            game = mod.BreakoutS(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
            game.fire_enabled = True
        elif (args.game=='BreakoutN'):
            mod = importlib.import_module('Breakout')
            game = mod.BreakoutN(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutSRA'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutSRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutSRAX'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutSRAExt(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutNRA'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutFNRA'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
            game.fire_enabled = True
            game.init_ball_speed_y = 0
        elif (args.game=='BreakoutBFNRA'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
            game.fire_enabled = True
        elif (args.game=='BreakoutNRA1'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRA(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename, RAenabled=False)
        elif (args.game=='BreakoutNRAX'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRAExt(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
        elif (args.game=='BreakoutBFNRAX'):
            mod = importlib.import_module('BreakoutRA')
            game = mod.BreakoutNRAExt(brick_rows=args.rows, brick_cols=args.cols, trainsessionname=trainfilename)
            game.fire_enabled = True
        elif (args.game=='Sapientino2'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=2)
        elif (args.game=='Sapientino2D'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=2)
            game.differential = True
        elif (args.game=='Sapientino2C'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=2)
            game.colorsensor = True
        elif (args.game=='Sapientino2DC'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=2)
            game.differential = True
            game.colorsensor = True
        elif (args.game=='Sapientino2X'):
            mod = importlib.import_module('Sapientino')
            game = mod.SapientinoExt(trainsessionname=trainfilename, nvisitpercol=2)
        elif (args.game=='Sapientino3'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=3)
        elif (args.game=='Sapientino3D'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=3)
            game.differential = True
        elif (args.game=='Sapientino3C'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=3)
            game.colorsensor = True
        elif (args.game=='Sapientino3DC'):
            mod = importlib.import_module('Sapientino')
            game = mod.Sapientino(trainsessionname=trainfilename, nvisitpercol=3)
            game.differential = True
            game.colorsensor = True
        elif (args.game=='Sapientino3X'):
            mod = importlib.import_module('Sapientino')
            game = mod.SapientinoExt(trainsessionname=trainfilename, nvisitpercol=3)
        else:
            print("ERROR: game %s not found." %args.game)
            sys.exit(1)
    except:
        print("ERROR: game %s not found." %args.game)
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
        elif (args.agent=='SarsaLin'):
            modname = 'RLAgent'
            mod = importlib.import_module(modname)
            agent = mod.SarsaAgent()
            agent.Qapproximation = True
        elif (args.agent=='MC'):
            modname = 'RLMCAgent'
            mod = importlib.import_module(modname)
            agent = mod.MCAgent()
        else:
            print("ERROR: agent %s not found." %modname)
            sys.exit(1)
    except:
        print("ERROR: agent %s not found." %modname)
        raise
        sys.exit(1)
    return agent

        
@atexit.register
def save():
    if game is not None and agent is not None and (not args.eval):
        # filename = trainfilename +"_%05d" % (self.iteration)
        filename = 'data/'+trainfilename
        savedata = [game.savedata(), agent.savedata()]
        np.savez(filename, gamedata = savedata[0], agentdata = savedata[1])
        print("Data saved successfully on file %s\n\n\n" %filename)

        
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
            game.loaddata(data['gamedata'])
            agent.loaddata(data['agentdata'])
        except:
            print("Can't load data from input file, wrong format.")
            #raise

            
def writeinfo(trainfilename,game,agent,init=True):
    global optimalPolicyFound
    infofile = open("data/"+trainfilename +".info","a+")
    allinfofile = open("data/all.info","a+")

    strtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())


    if (init):
        infofile.write("Date:    %s\n" %(strtime))
        infofile.write("Train:   %s\n" %(trainfilename))
        infofile.write("Game:    %s\n" %(args.game))
        infofile.write("Size:    %d x %d\n" %(args.rows, args.cols))
        infofile.write("Agent:   %s\n" %(agent.name))
        infofile.write("gamma:   %f\n" %(agent.gamma))
        infofile.write("epsilon: %f\n" %(agent.epsilon))
        infofile.write("alpha:   %f\n" %(agent.alpha))
        infofile.write("n-step:  %d\n" %(agent.nstepsupdates))
        infofile.write("lambda:  %f\n\n" %(agent.lambdae))
        infofile.write("\n%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae))
        #allinfofile.write("%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae))
    elif optimalPolicyFound:
        infofile.write("Optimal policy found.\n")
        infofile.write("goal reached:     %d\n" %(game.iteration))
        infofile.write("goal score:       %d\n" %(game.score))
        infofile.write("goal reward:      %.2f\n" %(game.cumreward))
        infofile.write("goal n. actions:  %d\n" %(game.numactions))
        infofile.write("highest reward:   %.2f\n" %(game.hireward))
        infofile.write("highest score:    %d\n" %(game.hiscore))
        infofile.write("elapsed time:     %d\n\n" %(game.elapsedtime))

        infofile.write("\n%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f,%d,%d,%.2f,%d,%.2f,%d,%d\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon, agent.alpha,agent.nstepsupdates,agent.lambdae,game.iteration,game.score,game.cumreward, game.numactions,game.hireward,game.hiscore,game.elapsedtime))
        allinfofile.write("%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f,%d,%d,%.2f,%d,%.2f,%d,%d\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon, agent.alpha,agent.nstepsupdates,agent.lambdae,game.iteration,game.score,game.cumreward, game.numactions,game.hireward,game.hiscore,game.elapsedtime))
    else:
        infofile.write("last iteration:   %d\n" %(game.iteration))
        infofile.write("last score:       %d\n" %(game.score))
        infofile.write("last reward:      %.2f\n" %(game.cumreward))
        infofile.write("last n. actions:  %d\n" %(game.numactions))
        infofile.write("highest reward:   %.2f\n" %(game.hireward))
        infofile.write("highest score:    %d\n" %(game.hiscore))
        infofile.write("elapsed time:     %d\n\n" %(game.elapsedtime))

        infofile.write("\n%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f,,,,,%d,%d,%.2f,%d,%.2f,%d,%d\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae, game.iteration,game.score,game.cumreward,game.numactions,game.hireward,game.hiscore,game.elapsedtime))
        allinfofile.write("%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f,,,,,%d,%d,%.2f,%d,%.2f,%d,%d\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae, game.iteration,game.score,game.cumreward,game.numactions,game.hireward,game.hiscore,game.elapsedtime))
    
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
def learn(game, agent, maxtime=-1, stopongoal=False):
    global optimalPolicyFound
    
    run = True
    last_goalreached = False
    next_optimal = False
    iter_goal = 0 # iteration in which first goal policy if found

    # timing the experiment
    exstart = time.time()
    elapsedtime0 = game.elapsedtime

    if (game.iteration>0 and not game.debug): # try an optimal run
        next_optimal = True
        game.iteration -= 1
        
    while (run and (args.niter<0 or game.iteration<=args.niter) and not game.userquit):

        game.reset() # increment game.iteration
        game.draw()
        time.sleep(game.sleeptime)
        if ((last_goalreached and agent.gamma==1) or next_optimal):
            agent.optimal = True
            next_optimal = False
        while (run and not game.finished):
            run = game.input()
            if game.pause:
                time.sleep(1)
                continue
            try:
                execution_step(game, agent)
                if (agent.error):
                    game.pause = True
                    agent.debug = True
                    agent.error = False
                game.draw()
                time.sleep(game.sleeptime)
            except KeyboardInterrupt:
                print("User quit")
                run = False

        # episode finished
        if (game.finished): 
            agent.notify_endofepisode(game.iteration)
            game.elapsedtime = (time.time() - exstart) + elapsedtime0
            game.print_report()
            time.sleep(game.sleeptime)

        # end of experiment
        if (agent.optimal and game.goal_reached()):
            optimalPolicyFound = True
            if (agent.gamma==1 or stopongoal):
                run = False
            #elif (iter_goal==0):
            #    iter_goal = game.iteration
            #elif (game.iteration>int(1.5*iter_goal)):
            #    run = False
        elif (maxtime>0 and game.elapsedtime > maxtime):
            run = False

        last_goalreached = game.goal_reached()

    if optimalPolicyFound:
        print("\n***************************")
        print("***  Goal policy found  ***")
        print("***************************\n")
        if (agent.Qapproximation):
            for a in range(0,game.nactions):
                print("Q[%d]" %a)
                print("       ",agent.Q[a].get_weights())

    exend = time.time()
    return  exend - exstart   # experiment time [seconds]




# evaluation process
def evaluate(game, agent, n): # evaluate best policy n times (no updates)
    i=0
    run = True
    game.sleeptime = 0.05
    if (game.gui_visible):
        game.sleeptime = 0.5
        game.pause = True
        
    while (i<n and run):
        game.reset()
        game.draw()
        time.sleep(game.sleeptime)

        agent.optimal = True
        while (run and not game.finished):
            run = game.input()
            if game.pause:
                time.sleep(1)
                continue
            execution_step(game, agent)
            game.draw()
            time.sleep(game.sleeptime)        
        game.print_report(printall=True)
        n=3
        i=0
        while (i<n):
            time.sleep(1)
            game.input()
            if game.pause:
                time.sleep(1)
            i += 1

        time.sleep(3)
        i += 1
    agent.optimal = False

    
# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RL games')
    parser.add_argument('game', type=str, help='game (e.g., Breakout)')
    parser.add_argument('agent', type=str, help='agent [Q, Sarsa, MC]')
    parser.add_argument('trainfile', type=str, help='file for learning strctures')
    parser.add_argument('-rows', type=int, help='number of rows [default: 3]', default=3)
    parser.add_argument('-cols', type=int, help='number of columns [default: 3]', default=3)
    parser.add_argument('-gamma', type=float, help='discount factor [default: 1.0]', default=1.0)
    parser.add_argument('-epsilon', type=float, help='epsilon greedy factor [default: -1 = adaptive]', default=-1)
    parser.add_argument('-alpha', type=float, help='alpha factor (-1 = based on visits) [default: -1]', default=-1)
    parser.add_argument('-nstep', type=int, help='n-steps updates [default: 1]', default=1)
    parser.add_argument('-lambdae', type=float, help='lambda eligibility factor [default: -1 (no eligibility)]', default=-1)
    parser.add_argument('-niter', type=float, help='stop after number of iterations [default: -1 = infinite]', default=-1)
    parser.add_argument('-maxtime', type=int, help='stop after maxtime seconds [default: -1 = infinite]', default=-1)
    parser.add_argument('--debug', help='debug flag', action='store_true')
    parser.add_argument('--gui', help='GUI shown at start [default: hidden]', action='store_true')
    parser.add_argument('--sound', help='Sound enabled', action='store_true')
    parser.add_argument('--eval', help='Evaluate best policy', action='store_true')
    parser.add_argument('--stopongoal', help='Stop experiment when goal is reached', action='store_true')
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
    if (args.debug):
        game.sleeptime = 1.0
        game.gui_visible = True
        
    agent.gamma = args.gamma
    agent.epsilon = args.epsilon
    agent.alpha = args.alpha
    agent.nstepsupdates = args.nstep
    agent.lambdae = args.lambdae
    agent.debug = args.debug

    game.init(agent)

    # load saved data
    load(trainfilename,game,agent)
    print("Game iteration: %d" %game.iteration)
    print("Game elapsedtime: %d" %game.elapsedtime)

    if (game.iteration==0):
        writeinfo(trainfilename,game,agent,init=True)

    # learning or evaluation process
    if (args.eval):
        evaluate(game, agent, 10)
    else:    
        et = learn(game, agent, args.maxtime, args.stopongoal)
        writeinfo(trainfilename,game,agent,init=False)

    print("Experiment terminated !!!\n")

    game.quit()

