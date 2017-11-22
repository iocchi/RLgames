#!/bin/env/python

import os

agent = 'Sarsa'
epsilon = 0.1
alpha = 0.01
niter = 10000
repetitions = 10

def Sapientino(nvisitpercol, lambdae, nstep)

game = 'Sapientino%d' %nvisitpercol
basetrainfilename = 'Sap%d_S_l0%d' %(nvisitpercol, int(lambdae*100)

for i in range(0,repetitions):
    cmd = "python game.py %s %s %s_%02d -epsilon %f -lambdae %f -alpha %f -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,niter)
    print cmd
    #os.system(cmd)

game = 'Sapientino2X'
basetrainfilename = 'Sap2X_S_e01_l09_a001'

for i in range(0,repetitions):
    cmd = "python game.py %s %s %s_%02d -epsilon %f -lambdae %f -alpha %f -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,niter)
    print cmd
    #os.system(cmd)

    
# Sapientino 2  - Sarsa(lambda)


# Sapientino 3  - Sarsa(lambda)
    
game = 'Sapientino3'
basetrainfilename = 'Sap3_S_e01_l09_a001'

for i in range(0,repetitions):
    cmd = "python game.py %s %s %s_%02d -epsilon %f -lambdae %f -alpha %f -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,niter)
    print cmd
    #os.system(cmd)

game = 'Sapientino3X'
basetrainfilename = 'Sap3X_S_e01_l09_a001'

for i in range(0,repetitions):
    cmd = "python game.py %s %s %s_%02d -epsilon %f -lambdae %f -alpha %f -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,niter)
    print cmd
    #os.system(cmd)

    
    
    
# Breakout 3x6  - Sarsa(lambda)
    
lambdae = 0.99    
game = 'BreakoutNRA'
basetrainfilename = 'B36_S_e01_l099_a001'

for i in range(0,repetitions):
    cmd = "python game.py %s %s %s_%02d -rows 3 -cols 6 -epsilon %f -lambdae %f -alpha %f -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,niter)
    print cmd
    #os.system(cmd)

lambdae = 0.99
game = 'BreakoutNRAX'
basetrainfilename = 'BX36_S_e01_l099_a001'

for i in range(0,repetitions):
    cmd = "python game.py %s %s %s_%02d -rows 3 -cols 6 -epsilon %f -lambdae %f -alpha %f -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,niter)
    print cmd
    #os.system(cmd)
    