#
#  This file is provided just for educational purposes 
#  and its development is not aligned with other code in this package.
#  
#  The RLAgent with proper parameter nstep should be used instead of this class.
#


#!/usr/bin/python

import pygame, sys
import numpy as np
import random
import time
import math
from math import fabs


class MCAgent(object):

    def __init__(self):
        self.command = 0       
        self.alpha = 0.5 # not used
        self.gamma = 1.0 
        self.epsilon = 0.5 
        self.optimal = False
        self.episode = []
        self.iteration = 0
        self.debug = False
        self.nstepsupdates = 0 # n-steps updates NOT USED HERE
        self.lambdae = -1 # lambda value for eligibility traces (-1 no eligibility)  NOT USED HERE
        self.name = 'RLMC'
        self.error = False

    def init(self, nstates, nactions):
        self.Q = np.zeros((nstates,nactions))
        # pi(a|x) non-normalized values (to normalize over all actions)
        self.pi = np.ones((nstates,nactions))
        self.Rsum = np.zeros((nstates,nactions))
        self.Rcnt = np.zeros((nstates,nactions))        
        self.nactions = nactions
        # temporary
        self.Rvisit = np.zeros((nstates,nactions))
        
    def reset(self):
        self.episode = []
        
    def set_action_names(self, an):
        self.action_names = an

        
    def savedata(self):
         return [self.Q, self.pi, self.Rsum, self.Rcnt]
         
    def loaddata(self, data):
         self.Q = data[0]
         self.pi = data[1]
         self.Rsum = data[2] 
         self.Rcnt = data[3] 


    def getQ(self, x, a):
        return self.Q[x,a]

    def getQA(self, x):
        return self.Q[x,:]

    def getpi(self, x, a):
        return self.pi[x,a]

    def getpiA(self, x):
        return self.pi[x,:]

    def getRavg(self, x, a):
        return float(self.Rsum[x,a])/self.Rcnt[x,a]

    def addR(self, x, a, r):
        self.Rsum[x,a] += r
        self.Rcnt[x,a] += 1

    def firstvisit(self, x, a):
        r = False
        if (self.Rvisit[x,a]==0):
            r = True
            self.Rvisit[x,a] = 1
        return r
        


    def decision(self, x):
        a = self.choose_action(x)
        if self.debug:
            print "Q: ",x," -> ",self.getQA(x)
            print "Decision: ",x,"  -> ",a

        return a
        
        
    def notify(self, x, a, r, x2):
        self.episode.append((x,a,r))
        

    def notify_endofepisode(self, iter):
        self.iteration = iter
        self.updateQ_episode()
        self.reset()

        
        
    def choose_action(self, x):  # choose action from state x
        
        if (self.optimal):  # executes best policy, no updates
            # Choose the action that maximizes expected reward.            
            piA = self.getpiA(x)
            a = np.argmax(piA)
            
        else:
            s = np.sum(self.getpiA(x)) 
            r = random.random() * s  # deal with non-normalized values
            
            if self.debug:
                print "pi(a|x) = ", self.getpiA(x)
                print "sum pi(a|x) ", s, "   - random: ", r
            
            
            c = 0
            a = 0
            while (c<s+1):
                c += self.getpi(x,a)
                if self.debug:
                    print "   - action ",a," p(a|x) = ",self.getpi(x,a)
                    #print "          rand ",r," < c ", c, "   "
                if (r<c):
                #    print "***"
                    break
                a += 1

        if self.debug:
            print "Action ",a 
            
        return a

        
    def rreturn(self, k):
        # return of current episode from state x_k
        r = 0
        g = 1.0
        while (k<len(self.episode)):
            ep = self.episode[k]
            r += g * ep[2]
            g = g * self.gamma
            k += 1
        return r

    def updateQ_episode(self):
    
        if (self.optimal):  # executes best policy, no updates
            return
        
        if (self.epsilon < 0):
            s = self.iteration
            k = 0.01 # decay weight 
            deltaS = 5000 # 0.5 value
            ee = math.exp(-k*(s-deltaS))
            epsilon = 0.9 * (1.0 - 1.0 / (1.0 + ee)) + 0.05
            #print "  -- iteration = ",s,"  -- epsilon = ",epsilon
        else:
            epsilon = self.epsilon

        
        self.Rvisit.fill(0) 
        # update Q for all state-action pairs in this episode
        k = 0
        for ep in self.episode:    
            x = ep[0] # current state
            a = ep[1] # current action
            if (self.firstvisit(x,a)):
                r = self.rreturn(k) # return from this state
                self.addR(x,a,r)
                self.Q[x,a] = self.getRavg(x,a)
            k += 1

        # update pi for all states in this episode
        for ep in self.episode:
            if (self.debug):
                print "[D] episode step ", ep

            x = ep[0] # current state

            ba = np.argmax(self.getQA(x))
            if (self.debug):
                print "[D] Q(x,:) = ", self.getQA(x)
                print "[D] best action ", ba
            
            # update pi epsilon-greedy
            for a in range(0,self.nactions):
                if (a==ba):
                    self.pi[x,a] = 1 - epsilon + epsilon / self.nactions
                else:
                    self.pi[x,a] = epsilon / self.nactions

            if (self.debug):
                print "[D] pi(a,:) = ", self.getpiA(x)

