#!/bin/env/python

import os
import time

def doExperiment(game, gameext, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, expid_from, exp_id_to):

    gameshortname = game[0]
    agentshortname = agent[0]
    gamecfg = '%s%s' %(game,gameext) 
    basetrainfilename = '%s%s_%s' %(gameshortname, gameext, agentshortname)
    if (gamma<1):
        basetrainfilename = basetrainfilename + '_g%03d' %(int(gamma*100))
    if (epsilon>0):
        basetrainfilename = basetrainfilename + '_e%02d' %(int(epsilon*100))
    if (lambdae>0):
        basetrainfilename = basetrainfilename + '_l%02d' %(int(lambdae*100))
    if (alpha>0):
        basetrainfilename = basetrainfilename + '_a%02d' %(int(alpha*100))
    if (nstep>1):
        basetrainfilename = basetrainfilename + '_n%d' %(nstep)

    str_stopongoal = ""
    if (stopongoal):
        str_stopongoal = "--stopongoal"

    for i in range(expid_from, exp_id_to+1):
        cmd = "python3 game.py %s %s %s_%02d -gamma %f -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d -maxtime %d %s" %(gamecfg,agent,basetrainfilename,i,gamma,epsilon,lambdae,alpha,nstep,niter,maxtime,str_stopongoal)
        print(cmd)
        os.system('xterm -geometry 100x20+0+20 -e "'+cmd+'"')
        # use -hold and & at the end for parallel execution and monitoring
        time.sleep(1)



agent = 'Sarsa'
gamma = 0.9
epsilon = 0.2
lambdae = -1
alpha = 0.1
nstep = 100
niter = -1
stopongoal = False
        
# Sapientino 2

maxtime = 120
nstep = 20

#doExperiment('Sapientino','2',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 3) 
#doExperiment('Sapientino','2C',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 5) 

maxtime = 2400

#doExperiment('Sapientino','2D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 3)
#doExperiment('Sapientino','2DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 6, 7) 

#doExperiment('Sapientino','2',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 3) 

#doExperiment('Sapientino','2C',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 

#doExperiment('Sapientino','2D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 3) 

#doExperiment('Sapientino','2DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 


# Sapientino 3

gamma = 0.9
epsilon = 0.2
alpha = 0.1
nstep = 20

maxtime = 300

# normal
doExperiment('Sapientino','3',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

# with options
doExperiment('Sapientino','3O',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 


gamma = 0.9
epsilon = 0.2
alpha = 0.1
nstep = 200

maxtime = 600

# differential
#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

# differential with options
#doExperiment('Sapientino','3DO',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 





#doExperiment('Sapientino','3Dx',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
#doExperiment('Sapientino','3DR',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
#doExperiment('Sapientino','3DxR',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 

#while maxtime<=1200:
#    doExperiment('Sapientino','3Dr',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 6) 
#    doExperiment('Sapientino','3Dxr',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 6) 
#    maxtime += 300


#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

gamma = 0.9
epsilon = 0.1
alpha = 0.1

stopongoal = False

maxtime = 1200

#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 3) 
#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

gamma = 0.99
epsilon = 0.1
alpha = 0.1

#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 
#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 

gamma = 0.99
epsilon = 0.1
alpha = 0.01


#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 4) 
#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 


