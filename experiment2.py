#!/bin/env/python

import os
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
    elif (epsilon==-1):
        basetrainfilename = basetrainfilename + '_eAi'
    elif (epsilon==-2):
        basetrainfilename = basetrainfilename + '_eAv'
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
        cmd = "python3 game.py %s %s %s_%02d -rows %d -cols %d -gamma %f -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d -maxtime %d %s" %(gamecfg,agent,basetrainfilename,i,rows,cols, gamma,epsilon,lambdae,alpha,nstep,niter,maxtime,str_stopongoal)
        xterm_cmd = 'xterm -geometry 100x20+0+20 -e "%s" ' %(cmd)
        # use -hold and & at the end for parallel execution and monitoring
        print(cmd)
        os.system(xterm_cmd)
        time.sleep(2)



agent = 'Sarsa'
gamma = 0.9999
epsilon = -2  # -1: auto on iterations, -2: auto on number of state visits
alpha = -1
lambdae = -1
niter = -1
stopongoal = False


# Breakout 4x4

rows = 4
cols = 4

nstep = 500
maxtime = 3000

doExperiment('Breakout','NRA', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3)
doExperiment('Breakout','NRAO', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 

nstep = 200
maxtime = 3000

doExperiment('Breakout','NRA', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
doExperiment('Breakout','NRAO', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 


#doExperiment('Breakout','NRARL', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 2) 
#doExperiment('Breakout','NRARLO', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 


# Breakout 4x5

rows = 4
cols = 5
nstep = 500
maxtime = 1200

#doExperiment('Breakout','NRARL', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
#doExperiment('Breakout','NRARLO', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 


# Breakout 4x6

rows = 4
cols = 6
nstep = 500
maxtime = 1200

#doExperiment('Breakout','NRARL', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
#doExperiment('Breakout','NRAORL', rows, cols, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 

# plot

def plotall():
    exp = ['B44NRA', 'B44NRARL']
    for e in exp:
      for i in range(3):
        cmd = 'python3 plotresults.py -datafiles data/%s_S_g0999_eAv_n200_%02d data/%sO_S_g0999_eAv_n200_%02d' %(e,i,e,i)
      os.system(cmd)    


#plotall()


