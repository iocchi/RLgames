import pygame, sys
import numpy as np
import random
import time
import math
from math import fabs


class RLAgent(object):

    def __init__(self):
        self.command = 0
        self.alpha = 0.5 # -1: adative
        self.gamma = 1.0
        self.epsilon = -1 #  -1: adaptive
        self.optimal = False
        self.episode = []
        self.iteration = 0
        self.debug = False
        self.name = 'RL'
        self.nstepsupdates = 0 # n-steps updates 
        self.lambdae = -1 # lambda value for eligibility traces (-1 no eligibility)

        
    def init(self, nstates, nactions):
        if nstates<100000:
            self.Q = np.zeros((nstates,nactions))
            self.Visits = np.zeros((nstates,nactions))
            self.sparse = False
        else:
            self.Q = {}
            self.Visits = {}
            self.sparse = True
        self.etraces = {} # eligibility traces map
        self.nactions = nactions

    def savedata(self):
         return [self.Q, self.Visits]
         
    def loaddata(self,data):
         self.Q = data[0]
         self.Visits = data[1]
         
        
    def getQ(self, x, a):
        if self.sparse:
            if x in self.Q:
                return self.Q[x][a]
            else:
                return 0
        else:
            return self.Q[x,a]

    def getQA(self, x):
        if self.sparse:
            if x in self.Q:
                return self.Q[x]
            else:
                return np.zeros(self.nactions)
        else:    
            return self.Q[x,:]
        
    def setQ(self, x, a, q):
        if self.sparse:
            if not x in self.Q:
                self.Q[x] = np.zeros(self.nactions)
            self.Q[x][a] = q
        else:
            self.Q[x,a] = q

    def addQ(self, x, a, q):
        if self.sparse:
            self.setQ(x,a,self.getQ(x,a)+q)
        else:
            self.Q[x,a] += q

    def setVisits(self, x, a, q):
        if self.sparse:
            if not x in self.Visits:
                self.Visits[x] = np.zeros(self.nactions)
            self.Visits[x][a] = q
        else:
            self.Visits[x,a] = q
        
    def incVisits(self, x, a):
        if self.sparse:
            self.setVisits(x,a,self.getVisits(x,a)+1)
        else:
            self.Visits[x,a] += 1
        # print "Visits ",x," <- ",self.Visits[x,:]

    def getVisits(self, x, a):
        if self.sparse:
            if x in self.Visits:
                return self.Visits[x][a]
            else:
                return 0
        else:
            return self.Visits[x,a]

    def getAlphaVisitsInc(self, x, a):
        self.incVisits(x,a)
        s = self.getVisits(x,a)
        a = 1.0/float(s)
        #print("visits: %d, a = %.6f" %(s,a))
        return a # math.sqrt(s)

    def getSumVisits(self, x):
        return np.sum(self.Visits[x,:])


    def choose_action(self, x):  # choose action from state x

        if (self.epsilon < 0):
            maxIter = 10000
            s = self.iteration #getSumVisits(x)
            p = min(float(s)/maxIter, 1.0)
            epsilon = 0.9 * (1.0 - p) + 0.1
            #print "  -- iter = ",s,"  -- epsilon = ",epsilon
        else:
            epsilon = self.epsilon
        
        if ((not self.optimal) and random.random()<epsilon):
            # Random action
            com_command = random.randint(0,self.nactions-1)
        else:
            # Choose the action that maximizes expected reward.            
            Qa = self.getQA(x)
            va = np.argmax(Qa)
            
            maxs = [i for i,v in enumerate(Qa) if v == va]
            if len(maxs) > 1:
                if self.command in maxs:
                    com_command = self.command
                elif self.optimal:
                    com_command = maxs[0]
                else:
                    com_command = random.choice(maxs)
            else:
                com_command = va

        return com_command

        
    def decision(self, x):
        
        a = self.choose_action(x)
        if self.debug:
            print "Q: ",x," -> ",self.getQA(x)
            print "Decision: ",x,"  -> ",a

        return a
        
    def notify(self, x, a, r, x2):

        #if (self.debug):
        #    print "Q update ",x," r: ",r
        
        self.episode.append((x,a,r))

#       ???       
#        if (abs(r)<0.0001): # r too small, no updates
#            return

        if (self.nstepsupdates<1):
            self.updateQ(x,a,r,x2)
        else:
            kn = len(self.episode) - self.nstepsupdates
            self.updateQ_n(kn,x2) # update state-action n-steps back

    def notify_endofepisode(self, iter):
        self.iteration = iter
        if (self.nstepsupdates>0):
            kn = max(0,len(self.episode) - self.nstepsupdates)
            while (kn < len(self.episode)):
                self.updateQ_n(kn,None) # update state-action n-steps back
                kn += 1
        self.episode = []
        #print "reset e"
        self.etraces = {} # eligibility taces map


    def getActionValue(self, x2):
        print("ERROR: function getActionValue not implemented")
        return 0


    def setEligibility(self, x, a):    
        # update eligibility values of current (x,a)
        # put to zero eligibility for all actions from this state
        for ai in range(self.nactions):
            if (ai!=a):
                self.etraces.pop((x,ai),None)
        accumulating_traces = False # False for replacing traces (more stable)
        if ((x,a) in self.etraces and accumulating_traces):
            self.etraces[(x,a)] += 1
        else:
            self.etraces[(x,a)] = 1
        toremove = [] # remove close-to-zero elements        
        for e in self.etraces:
            # update eligibility values
            self.etraces[e] *= self.gamma * self.lambdae
            if (self.etraces[e]<0.001): # remove close-to-zero elements
                toremove.append(e)
        # remove close-to-zero elements
        for e in toremove:
            self.etraces.pop(e)
        if (self.debug):
            print(" etraces: %d " %(len(self.etraces)))


    def updateEligibility(self, x, a, alpha, delta):
    
        if (self.debug):
            print("updating e: %d %d ..." %(x,a))
            print(" etraces: %d " %(len(self.etraces)))
        for e in self.etraces:
            # update Q table
            if (delta!=0):
                if (alpha<0):
                    alpha = self.getAlphaVisitsInc(e[0],e[1])
                q = alpha * delta * self.etraces[e]
                self.addQ(e[0],e[1],q)
                if (self.debug):
                    print "  -- e ",e," ",self.etraces[e]
                    print "  -- e x:",e[0]," a:",e[1]
                    print "  -- alpha: ",alpha,"  delta: ", delta
                    print "  -- Q(e) = ", self.getQ(e[0],e[1])
        if (self.debug):
            print "\n"
        # clear traces after update
        #self.etraces = {}

            
            
    def updateQ(self,x,a,r,x2):
    
        if (self.optimal):  # executes best policy, no updates
            return

        if (self.lambdae>0):
            self.setEligibility(x,a)

        if (abs(r)<1e-3): # r too small, no updates
            return


        # Q of current state
        prev_Q = self.getQ(x,a)

        vQa = self.getActionValue(x2)
        
            
        delta = r + self.gamma * vQa - prev_Q

        if (self.debug):
            print ' == ',x,' A: ',a,' -> r: ',r,' -> ',x2,' prev_Q: ', prev_Q, '  vQa: ', vQa
            print ' == Q update Q ',x,',',a,' <-  ...  Q ',x2,' = ', vQa
        
        if (self.lambdae>0):
            self.updateEligibility(x,a,self.alpha,delta)
        else:
            if (self.alpha>=0):
                alpha = self.alpha
            else:
                alpha = self.getAlphaVisitsInc(x,a)
            q = alpha * delta 
            self.addQ(x,a,q)
        

    def rreturn(self, k, n):
        # n-steps return of current episode from state x_k 
        r = 0
        g = 1.0
        l = min(len(self.episode), k+n)
        while (k<l):
            ep = self.episode[k]
            r += g * ep[2]
            g = g * self.gamma
            k += 1
        return r

        
    def updateQ_n(self,kn,x2): # n-steps Q update
        # kn = index of state n-steps back
        # x2 = next state after last action

        if (self.debug):
            print "updateQ_n ",kn, " optimal = ",self.optimal
        
        if (self.optimal):  # executing best policy, no updates
            return

        if (self.debug):
            print "debug updateQ_n ... ",kn

        if (kn<0):  # kn not valid
            return

        ep = self.episode[kn]
        x_kn = ep[0]
        a_kn = ep[1]
        g = self.rreturn(kn, self.nstepsupdates) # n-steps return from state x_{kn}

        if (self.lambdae>0):
            self.setEligibility(x_kn,a_kn)

        if (self.debug):
            print "return = ",g

        # if not at the end of the episode
        if (not x2 is None):
            g += math.pow(self.gamma, self.nstepsupdates) * self.getActionValue(x2) # expected value in next state

        delta = (g - self.getQ(x_kn,a_kn))

        if (self.debug):
            print "debug updateQ_n ... ",kn
            print "x = ",x_kn, "  a = ",a_kn
            print "return = ",g
            print "delta = ",delta

        if (self.lambdae>0):
            self.updateEligibility(x_kn,a_kn,self.alpha,delta)
        else:
            if (self.alpha>=0):
                alpha = self.alpha
            else:
                alpha = self.getAlphaVisitsInc(x_kn,a_kn)
            q = alpha * delta    
            self.addQ(x_kn,a_kn,q)



class QAgent(RLAgent):

    def __init__(self):
        RLAgent.__init__(self)
        self.name = 'Q-Learning'

    def getActionValue(self, x2):
        # Q-learning
        maxQa = max(self.getQA(x2)) 
        return maxQa


class SarsaAgent(RLAgent):

    def __init__(self):
        RLAgent.__init__(self)
        self.name = 'Sarsa'

    def getActionValue(self, x2):
        # Sarsa
        sarsa_a = self.choose_action(x2)
        sarsaQa = self.getQ(x2,sarsa_a) 
        return sarsaQa
        
