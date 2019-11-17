#!/bin/env/python

import os
import time

def doExperiment(game, gameext, agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, exp_seeds):

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

    for i in exp_seeds:
        cmd = "python3 game.py %s %s %s_%03d -seed %d -gamma %f -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d -maxtime %d %s" %(gamecfg,agent,basetrainfilename,i,i, gamma,epsilon,lambdae,alpha,nstep,niter,maxtime,str_stopongoal)
        print(cmd)
        os.system('xterm -geometry 100x20+0+20 -e "'+cmd+'"')
        # use -hold and & at the end for parallel execution and monitoring
        time.sleep(1)



agent = 'Sarsa'
gamma = 0.99
epsilon = -2
lambdae = -1
alpha = -1
niter = -1
stopongoal = False
        
# Sapientino 2


nstep = 20

maxtime = 120

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


nstep = 20

maxtime = 200

# normal
doExperiment('Sapientino','3',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, [101,102,103]) 

# with options
doExperiment('Sapientino','3O',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, [101, 102, 103]) 


nstep = 20

maxtime = 600

# differential
#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, [101,102,103]) 

# differential with options
#doExperiment('Sapientino','3DO',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, [101,102,103]) 





#doExperiment('Sapientino','3Dx',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
#doExperiment('Sapientino','3DR',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 
#doExperiment('Sapientino','3DxR',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 3) 

#while maxtime<=1200:
#    doExperiment('Sapientino','3Dr',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 6) 
#    doExperiment('Sapientino','3Dxr',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 6) 
#    maxtime += 300


#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 


maxtime = 1200

#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 2, 3) 
#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 


#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 
#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 



#doExperiment('Sapientino','3D',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 4, 4) 
#doExperiment('Sapientino','3DC',agent, gamma, epsilon, lambdae, alpha, nstep, niter, maxtime, stopongoal, 1, 1) 


