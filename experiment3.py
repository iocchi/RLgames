#!/bin/env/python

import os
import thread
import time

def doExperiment(game, gameext, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, expid_from, exp_id_to):

    gameshortname = game[0]
    agentshortname = agent[0]
    gamecfg = '%s%s' %(game,gameext) 
    basetrainfilename = '%s%s_%s' %(gameshortname, gameext, agentshortname)
    if (gamma<1):
        basetrainfilename = basetrainfilename + '_g%04d' %(int(gamma*1000))
    if (epsilon>0):
        basetrainfilename = basetrainfilename + '_e%03d' %(int(epsilon*100))
    if (lambdae>0):
        basetrainfilename = basetrainfilename + '_l%03d' %(int(lambdae*100))
    if (alpha>0):
        basetrainfilename = basetrainfilename + '_a%03d' %(int(alpha*100))
    if (nstep>1):
        basetrainfilename = basetrainfilename + '_n%d' %(nstep)

    str_stopongoal = ""
    if (stopongoal):
        str_stopongoal = "--stopongoal"

    for i in range(expid_from, exp_id_to+1):
        cmd = "python2 game.py %s %s %s_%02d -gamma %f -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d -maxtime %d %s" %(gamecfg,agent,basetrainfilename,i,gamma,epsilon,lambdae,alpha,nstep,niter,maxtime,str_stopongoal)
        xterm_cmd = 'xterm -geometry 100x20+0+20 -e "%s" ' %(cmd)
        # use -hold and & at the end for parallel execution and monitoring
        print cmd
        os.system(cmd)
        #thread.start_new_thread( os.system, (cmd,) )
        print('sleep...')
        time.sleep(3)



agent = 'Sarsa'
gamma = 0.999
epsilon = 0.2
lambdae = -1
alpha = -1
niter = -1
stopongoal = False

# main

nstep = 20

maxtime = 1200

#doExperiment('Minecraft','',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

maxtime = 1800

doExperiment('Minecraft','D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 4) 

#doExperiment('Minecraft','D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

#nstep = 50
#doExperiment('Minecraft','',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 5) 

#nstep = 100
#doExperiment('Minecraft','',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 5) 
