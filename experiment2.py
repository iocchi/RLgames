#!/bin/env/python

import os
import thread
import time

def doExperiment(game, gameext, rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, expid_from, exp_id_to):

    gameshortname = game[0]
    agentshortname = agent[0]
    gamecfg = '%s%s' %(game,gameext) 
    basetrainfilename = '%s%d%d%s_%s' %(gameshortname, rows, cols, gameext, agentshortname)
    if (gamma<1):
        basetrainfilename = basetrainfilename + '_g%04d' %(int(gamma*1000))
    if (epsilon>0):
        basetrainfilename = basetrainfilename + '_e%03d' %(int(epsilon*100))
    if (lambdae>0):
        basetrainfilename = basetrainfilename + '_l%03d' %(int(lambdae*100))
    if (alpha>0):
        basetrainfilename = basetrainfilename + '_a%03d' %(int(alpha*100))
    if (nstep>1):
        basetrainfilename = basetrainfilename + '_n%03d' %(nstep)
    str_stopongoal = ""
    if (stopongoal):
        str_stopongoal = "--stopongoal"

    for i in range(expid_from, exp_id_to+1):
        cmd = "python game.py %s %s %s_%02d -rows %d -cols %d -gamma %f -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d -maxtime %d %s" %(gamecfg,agent,basetrainfilename,i,rows,cols, gamma,epsilon,lambdae,alpha,nstep,niter,maxtime,str_stopongoal)
        xterm_cmd = 'xterm -geometry 100x20+0+20 -e "%s" ' %(cmd)
        # use -hold and & at the end for parallel execution and monitoring
        print cmd
        os.system(cmd)
        #thread.start_new_thread( os.system, (cmd,) )
        time.sleep(2)



agent = 'Sarsa'
gamma = 0.99
epsilon = 0.2
lambdae = -1

nstep = 200
niter = -1
stopongoal = False


# Breakout 4x4

rows = 4
cols = 4

gamma = 0.999
nstep = 200
alpha = -1


# range of experiments
idfrom = 31
idto = 31

for i in range(1,11):
    maxtime = 600*i
    doExperiment('Breakout','NDNRA', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, idfrom, idto) 

#doExperiment('Breakout','NRAX', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, idfrom, idto) 

maxtime = 180

#doExperiment('Breakout','BFNRA', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, idfrom, idto) 
#doExperiment('Breakout','BFNRAX', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, idfrom, idto) 


