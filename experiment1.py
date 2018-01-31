#!/bin/env/python

import os
import thread
import time

def doExperiment(game, gameext, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, expid_from, exp_id_to):

    gameshortname = game[0]
    agentshortname = agent[0]
    gamecfg = '%s%s' %(game,gameext) 
    basetrainfilename = '%s%s_%s' %(gameshortname, gameext, agentshortname)
    if (gamma<1):
        basetrainfilename = basetrainfilename + '_g0%d' %(int(gamma*100))
    if (epsilon>0):
        basetrainfilename = basetrainfilename + '_e0%d' %(int(epsilon*100))
    if (lambdae>0):
        basetrainfilename = basetrainfilename + '_l0%d' %(int(lambdae*100))
    if (alpha>0):
        basetrainfilename = basetrainfilename + '_a0%d' %(int(alpha*100))
    if (nstep>1):
        basetrainfilename = basetrainfilename + '_n%d' %(nstep)

    for i in range(expid_from, exp_id_to+1):
        cmd = "xterm -geometry 100x20+0+20 -e \"python game.py %s %s %s_%02d -gamma %f -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d -maxtime %d\" " %(gamecfg,agent,basetrainfilename,i,gamma,epsilon,lambdae,alpha,nstep,niter,maxtime)
        # use -hold and & at the end for parallel execution and monitoring
        print cmd
        os.system(cmd)
        #thread.start_new_thread( os.system, (cmd,) )
        time.sleep(1)



agent = 'Sarsa'
gamma = 0.9
epsilon = 0.2
lambdae = -1
alpha = 0.1
nstep = 100
niter = -1

        
# main

maxtime = 500

#doExperiment('Sapientino','2',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 4, 5) 

#doExperiment('Sapientino','2C',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 4, 5) 


maxtime = 1200

doExperiment('Sapientino','2D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 6, 7) 

doExperiment('Sapientino','2DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 6, 7) 

agent = 'Q'

#doExperiment('Sapientino','2',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 2, 3) 

#doExperiment('Sapientino','2C',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 1, 3) 

#doExperiment('Sapientino','2D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 2, 3) 

#doExperiment('Sapientino','2DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, 1, 3) 



