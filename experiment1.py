#!/bin/env/python

import os

agent = 'Sarsa'
epsilon = 0.1
alpha = 0.01
niter = 10000
repetitions = 1

def ExpSapientino(nvisitpercol, strextended, lambdae, nstep):

    game = 'Sapientino%d%s' %(nvisitpercol, strextended) 
    basetrainfilename = 'Sap%d%s_S' %(nvisitpercol, strextended) 
    if (lambdae>0):
        basetrainfilename = basetrainfilename + '_l0%d' %(int(lambdae*100))
    if (nstep>1):
        basetrainfilename = basetrainfilename + '_n%d' %(nstep)

    for i in range(0,repetitions):
        cmd = "python game.py %s %s %s_%02d -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d" %(game,agent,basetrainfilename,i,epsilon,lambdae,alpha,nstep,niter)
        print cmd
        os.system(cmd)



        
def ExpBreakout(rows, cols, strextended, lambdae, nstep):

    game = 'BreakoutNRA%s' %(strextended) 
    basetrainfilename = 'BNRA%s_S' %(strextended) 
    if (lambdae>0):
        basetrainfilename = basetrainfilename + '_l0%d' %(int(lambdae*100))
    if (nstep>1):
        basetrainfilename = basetrainfilename + '_n%d' %(nstep)

    for i in range(0,repetitions):
        cmd = "python game.py %s %s %s_%02d -rows %d -cols %d -epsilon %f -lambdae %f -alpha %f -nstep %d -niter %d" %(game,agent,basetrainfilename,i,rows,cols,epsilon,lambdae,alpha,nstep,niter)
        print cmd
        os.system(cmd)



        
# main

ExpSapientino(2,'',0.0,1) # Sarsa
ExpSapientino(2,'X',0.0,1) # Sarsa
ExpSapientino(2,'',0.9,1) # Sarsa(\lambda)
ExpSapientino(2,'X',0.9,1) # Sarsa(\lambda)
ExpSapientino(2,'',-1,10) # n-step Sarsa
ExpSapientino(2,'X',-1,10) # n-step Sarsa

ExpSapientino(3,'',0.0,1) # Sarsa
ExpSapientino(3,'X',0.0,1) # Sarsa
ExpSapientino(3,'',0.9,1) # Sarsa(\lambda)
ExpSapientino(3,'X',0.9,1) # Sarsa(\lambda)
ExpSapientino(3,'',-1,10) # n-step Sarsa
ExpSapientino(3,'X',-1,10) # n-step Sarsa

ExpBreakout(3,6,'',0.99,1) # Sarsa(\lambda)
ExpBreakout(3,6,'X',0.99,1) # Sarsa(\lambda)
