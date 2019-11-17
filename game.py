#!/usr/bin/python

import os, signal
import importlib
import pygame, sys
import numpy as np
import atexit
import math, time
from math import fabs
from time import gmtime, strftime
import argparse

np.set_printoptions(precision=3)

trainfilename = 'default'
optimalPolicyFound = False 

args = None
game = None
agent = None


GAMES = {
    'SimpleGrid':  [ "importlib.import_module('SimpleGrid').SimpleGrid", None ],

    'BreakoutS':   [ "importlib.import_module('Breakout').BreakoutS", None ],
    'BreakoutSO':   [ "importlib.import_module('Breakout').BreakoutS",
                     "game.RA_exploration_enabled = True" ],

    'BreakoutFS':  [ "importlib.import_module('Breakout').BreakoutS", 
                     "game.fire_enabled = True" ],
    'BreakoutN':   [ "importlib.import_module('Breakout').BreakoutN", None ],

    'BreakoutSRA':   [ "importlib.import_module('BreakoutRA').BreakoutSRA", None ],
    'BreakoutSRAO':   [ "importlib.import_module('BreakoutRA').BreakoutSRA", 
                        "game.RA_exploration_enabled = True" ],

    'BreakoutSRAX':   [ "importlib.import_module('BreakoutRA').BreakoutSRAExt", None ],

    'BreakoutNRA':   [ "importlib.import_module('BreakoutRA').BreakoutNRA", None ],
    'BreakoutNRARL':   [ "importlib.import_module('BreakoutRA').BreakoutNRA", 
                          "game.RA.left_right=False"  ],

    'BreakoutNRAO':   [ "importlib.import_module('BreakoutRA').BreakoutNRA", 
                        "game.RA_exploration_enabled = True"  ],
    'BreakoutNRARLO':   [ "importlib.import_module('BreakoutRA').BreakoutNRA", 
                          "game.RA.left_right=False\ngame.RA_exploration_enabled = True"  ],

    'BreakoutNDNRA':   [ "importlib.import_module('BreakoutRA').BreakoutNRA",
                         "game.deterministic = False" ],
    'BreakoutFNRA':   [ "importlib.import_module('BreakoutRA').BreakoutNRA",
                         "game.fire_enabled = True" ],  
    'BreakoutNRAX':   [ "importlib.import_module('BreakoutRA').BreakoutNRAExt", None ],  
    'BreakoutFNRAX':   [ "importlib.import_module('BreakoutRA').BreakoutNRAExt",
                         "game.fire_enabled = True" ],  
    'Sapientino2':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=2" ],  
    'Sapientino2D':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=2\ngame.differential = True\n" ],  
    'Sapientino2C':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=2\ngame.colorsensor = True\n" ],  
    'Sapientino2DC':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=2\ngame.differential = True\ngame.colorsensor = True\n" ],  
    'Sapientino2X':   [ "importlib.import_module('Sapientino').SapientinoExt", 
                       "game.nvisitpercol=2" ],  

    'Sapientino3':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3" ],  
    'Sapientino3D':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\n" ],  
    'Sapientino3DO':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.RA_exploration_enabled = True\n" ],  


    'Sapientino3C':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.colorsensor = True\n" ],  
    'Sapientino3DC':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.colorsensor = True\n" ],  
    'Sapientino3O':   [ "importlib.import_module('Sapientino').Sapientino", 
                        "game.nvisitpercol=3\ngame.RA_exploration_enabled = True" ],  
    'Sapientino3X':   [ "importlib.import_module('Sapientino').SapientinoExt", 
                       "game.nvisitpercol=3" ],  
    'Sapientino3DR':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.RA.reward_shaping_enabled = True" ],  
    'Sapientino3Dr':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.reward_shaping_enabled = True" ],  
    'Sapientino3Dx':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.RA_exploration_enabled = True" ],  
    'Sapientino3Dxr':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.RA_exploration_enabled = True\ngame.reward_shaping_enabled = True" ],  
    'Sapientino3DxR':   [ "importlib.import_module('Sapientino').Sapientino", 
                       "game.nvisitpercol=3\ngame.differential = True\ngame.RA_exploration_enabled = True\ngame.RA.reward_shaping_enabled = True" ],  

    'Minecraft':   [ "importlib.import_module('Minecraft').Minecraft", None], 
    'MinecraftO':   [ "importlib.import_module('Minecraft').Minecraft", 
                      "game.RA_exploration_enabled = True" ], 
    'MinecraftD':   [ "importlib.import_module('Minecraft').Minecraft", 
                      "game.differential = True" ], 
    'MinecraftDO':   [ "importlib.import_module('Minecraft').Minecraft", 
                      "game.differential = True\ngame.RA_exploration_enabled = True" ], 


    'CP':   [ "importlib.import_module('CocktailParty').CocktailParty", None], 
    'CP1':   [ "importlib.import_module('CocktailParty').CocktailParty", 
               "game.setOneTask()"], 
    'CPx':   [ "importlib.import_module('CocktailParty').CocktailParty", 
               "game.RA_exploration_enabled = True"], 
    'CPD':   [ "importlib.import_module('CocktailParty').CocktailParty", 
               "game.differential = True"], 
    'CPdyn':   [ "importlib.import_module('CocktailPartyDynamic').CocktailParty", 
               "game.differential = False"], 
    'CPros':   [ "importlib.import_module('CocktailPartyROS').CocktailParty", 
               "game.differential = False"], 


    'PP':   [ "importlib.import_module('PickAndPlace').PickAndPlace", None], 
    'PPx':   [ "importlib.import_module('PickAndPlace').PickAndPlace", 
               "game.RA_exploration_enabled = True" ], 

}



def loadGameModule():
    print("Loading game %s" %args.game)
    try:
        # default sizes
        if 'Sapientino' in args.game:
            args.rows = 5 
            args.cols = 7
        elif 'Minecraft' in args.game:
            args.rows = 10 
            args.cols = 10
        game = eval(GAMES[args.game][0])(args.rows, args.cols, trainfilename)
        if GAMES[args.game][1] is not None:
            exec(GAMES[args.game][1])

        if True:
            pass
        elif (args.game=='CPd'):
            mod = importlib.import_module('CocktailParty')
            game = mod.CocktailParty(trainsessionname=trainfilename,rows=args.rows,cols=args.cols)
            game.differential = True
        elif (args.game=='CPD'):
            mod = importlib.import_module('CocktailPartyDynamic')
            game = mod.CocktailParty(trainsessionname=trainfilename)
            game.differential = False
        elif (args.game=='CPR'):
            mod = importlib.import_module('CocktailPartyROS')
            game = mod.CocktailParty(trainsessionname=trainfilename)
        elif (args.game=='PP'):
            mod = importlib.import_module('PickAndPlace')
            game = mod.PickAndPlace(trainsessionname=trainfilename)
        elif (args.game=='PPx'):
            mod = importlib.import_module('PickAndPlace')
            game = mod.PickAndPlace(trainsessionname=trainfilename)
            game.RA_exploration_enabled = True
        else:
            print("ERROR: game %s not found." %args.game)
            sys.exit(1)
    except:
        print("ERROR: game %s not found." %args.game)
        raise
        sys.exit(1)
    return game



AGENTS = {

    'Q': [ "importlib.import_module('RLAgent').QAgent", None ],
    'Sarsa': [ "importlib.import_module('RLAgent').SarsaAgent", None ],
    'SarsaLin': [ "importlib.import_module('RLAgent').SarsaAgent", 
                  "agent.Qapproximation = True" ],
    'MC': [ "importlib.import_module('RLMCAgent').MCAgent", None ],
    'RMax': [ "importlib.import_module('RMaxAgent').RMaxAgent", None ],
}


def loadAgentModule():
    print("Loading agent "+args.agent)
    try:
        agent = eval(AGENTS[args.agent][0])()
        if AGENTS[args.agent][1] is not None:
            exec(AGENTS[args.agent][1])
    except:
        print("ERROR: agent %s not found." %args.agent)
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
        data = np.load(fn, allow_pickle=True)  # for Python3
        s = "Data loaded from " + fn + " successfully."
        print(s)
    except IOError:
        s = "Error: can't find file or read data from file " + fn +" -> initializing new structures"
        print(s)

    if data is not None:
        try:
            game.loaddata(data['gamedata'])
            agent.loaddata(data['agentdata'])
        except Exception as e:
            print(e)
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
        infofile.write("\n%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%.3f\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae))
        #allinfofile.write("%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%f\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon,agent.alpha,agent.nstepsupdates,agent.lambdae))
    else:
        infofile.write("iteration:        %d\n" %(game.iteration))
        infofile.write("goal score:       %d\n" %(game.score))
        infofile.write("goal reward:      %.2f\n" %(game.cumreward))
        infofile.write("goal n. actions:  %d\n" %(game.numactions))
        infofile.write("highest reward:   %.2f\n" %(game.hireward))
        infofile.write("highest score:    %d\n" %(game.hiscore))
        infofile.write("elapsed time:     %d\n" %(game.elapsedtime))

        if optimalPolicyFound:
            infofile.write("Optimal policy found.\n")

        try:
            infofile.write("\n"+game.report_str+"\n")
        except:
            pass

        infofile.write("\n%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%.3f,%d,%d,%.2f,%d,%.2f,%d,%d\n\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon, agent.alpha,agent.nstepsupdates,agent.lambdae,game.iteration,game.score,game.cumreward, game.numactions,game.hireward,game.hiscore,game.elapsedtime))
        allinfofile.write("%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%d,%.3f,%d,%d,%.2f,%d,%.2f,%d,%d\n" %(strtime,trainfilename,args.game,args.rows,args.cols,agent.name,agent.gamma,agent.epsilon, agent.alpha,agent.nstepsupdates,agent.lambdae,game.iteration,game.score,game.cumreward, game.numactions,game.hireward,game.hiscore,game.elapsedtime))
    
    infofile.flush()
    infofile.close()
    allinfofile.flush()
    allinfofile.close()

def handler(signum, frame):
    global userquit
    print('User quit (CTRL-C) [signal: %d]' %signum)
    userquit = True


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
    global optimalPolicyFound, userquit
    
    run = True
    userquit = False
    last_goalreached = False
    next_optimal = False
    iter_goal = 0 # iteration in which first goal policy if found

    # timing the experiment
    exstart = time.time()
    elapsedtime0 = game.elapsedtime

    if (maxtime>0 and game.elapsedtime >= maxtime):
        run = False
    #elif (game.iteration>0 and game.iteration<100 and not game.debug): # try an optimal run  ???
    #    next_optimal = True
    #    game.iteration -= 1

    while (run and (args.niter<0 or game.iteration<=args.niter)):
        game.reset() # increment game.iteration
        game.draw()
        time.sleep(game.sleeptime)
        if ((last_goalreached and agent.gamma==1) or next_optimal):
            agent.optimal = True
            next_optimal = False
        while (run and not game.finished):
            grun = game.input()
            if (not grun):
                userquit = True
            if game.pause:
                time.sleep(1)
                continue

            execution_step(game, agent)

            if (agent.error):
                game.pause = True
                agent.debug = True
                agent.error = False            
            game.draw()
            time.sleep(game.sleeptime)

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
        elif (maxtime>0 and game.elapsedtime >= maxtime):
            run = False
        elif (userquit or game.userquit):
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
    

# evaluation process
def evaluate(game, agent, n): # evaluate best policy n times (no updates)
    i=0
    run = True
    game.sleeptime = 0.001
    if (game.gui_visible):
        game.sleeptime = 0.1
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
        if (game.gui_visible):
            n=3
            j=0
            while (j<n):
                time.sleep(1)
                game.input()
                if game.pause:
                    time.sleep(1)
                j += 1
            time.sleep(3)
        i += 1
    agent.optimal = False



    
# main
if __name__ == "__main__":


    # Set the signal handler
    signal.signal(signal.SIGINT, handler)

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
    parser.add_argument('-seed', type=int, help='random seed [default: -1 = do no set]', default=-1)
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

    if args.seed>0:
        agent.setRandomSeed(args.seed)
        game.setRandomSeed(args.seed)

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
        learn(game, agent, args.maxtime, args.stopongoal)
        writeinfo(trainfilename,game,agent,init=False)

    print("Experiment terminated after iteration: %d!!!\n" %game.iteration)
    #print('saving ...')
    #save()
    print('Game over')
    game.quit()

