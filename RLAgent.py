import pygame, sys
import numpy as np
import random
import time
import math
from math import fabs


class RLAgent(object):

    def __init__(self):
        self.command = 0
        self.gamma = 1.0
        self.epsilon = -1 #  -1: adaptive
        self.alpha = 0.5 # -1: adative
        self.nstepsupdates = 0 # n-steps updates 
        self.lambdae = -1 # lambda value for eligibility traces (-1 no eligibility)
        self.optimal = False
        self.option_enabled = False # whether to exploit options
        self.episode = []  # list of (x,a,r) gathered during episode
        self.iteration = 0
        self.debug = False
        self.name = 'RL'
        self.sparse = False
        self.Qapproximation = False
        self.error = False
        self.SA_failure = [] # memory state-action failures
        
    def init(self, nstates, nactions):
        if (self.Qapproximation):
            try:
                from keras.models import Sequential
                from keras.layers import Dense, Activation
            except:
                print('Install keras if you want to use function approximation')

            self.Q = {}
            for a in range(0,nactions):
                self.Q[a] = Sequential()
                self.Q[a].add(Dense(15, input_dim=2))
                self.Q[a].add(Activation('sigmoid'))
                self.Q[a].add(Dense(1))
                self.Q[a].add(Activation('linear'))
                self.Q[a].compile(loss='mse', optimizer='sgd')
            self.Visits = {}
        elif nstates<10000:
            self.Q = np.zeros((nstates,nactions))
            self.Visits = np.zeros((nstates,nactions))
            self.sparse = False
        else:
            self.Q = {}
            self.Visits = {}
            self.sparse = True

        self.etraces = {} # eligibility traces map
        self.nactions = nactions
        
        print("Agent: %s" %self.name)
        print("  gamma: %f" %self.gamma)
        print("  epsilon: %f" %self.epsilon)
        print("  alpha: %f" %self.alpha)
        print("  nsteps: %d" %self.nstepsupdates)
        print("  lambda: %f" %self.lambdae)

    def set_action_names(self, an):
        self.action_names = an

    def setRandomSeed(self,seed):
        random.seed(seed)
        np.random.seed(seed)

    def savedata(self):
        return [self.Q, self.Visits, self.SA_failure,random.getstate(),np.random.get_state()]


    def loaddata(self,data):
        self.Q = data[0]
        self.Visits = data[1]
        self.SA_failure = data[2]
        if (len(data)>3):
            print('Set random generator state from file.')
            random.setstate(data[3])
            np.random.set_state(data[4])       

    def getQ(self, x, a):
        if (self.Qapproximation):
            xa = np.zeros((1,2))
            xa[(0,0)] = x
            xa[(0,1)] = 1
            vQaa = self.Q[a].predict(xa) 
            vQa = vQaa[0][0]
            #print(" Q[x] predict  %s\n" %vQa)
            return vQa
        elif self.sparse:
            if x in self.Q:
                return self.Q[x][a]
            else:
                return 0
        else:
            return self.Q[x,a]

    def getQA(self, x):
        if (self.Qapproximation):
            r = []
            for a in range(0,self.nactions):
                r.append(self.getQ(x,a))
            return r
        elif self.sparse:
            if x in self.Q:
                return self.Q[x]
            else:
                return np.zeros(self.nactions)
        else:    
            return self.Q[x,:]
        
    def setQ(self, x, a, q):
        if (self.Qapproximation):
            xa = np.zeros((1,2))
            qa = np.zeros((1,1))
            xa[(0,0)] = x/10
            xa[(0,1)] = x%10
            qa[(0,0)] = q
            self.Q[a].fit(xa,qa,verbose=0)
        elif self.sparse:
            if not x in self.Q:
                self.Q[x] = np.zeros(self.nactions)
            self.Q[x][a] = q
            #if (q<0):
            #    print("Neg Q [%d,%d] <- %.3f" %(x,a,q))
            #    self.error = True # signal error
        else:
            self.Q[x,a] = q

    def addQ(self, x, a, q):
        if self.sparse or self.Qapproximation:
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
        if (self.Qapproximation):
            self.Visits[x,a] += 1
        elif self.sparse:
            self.setVisits(x,a,self.getVisits(x,a)+1)
        else:
            self.Visits[x,a] += 1
        # print("Visits ",x," <- ",self.Visits[x,:])

    def getVisits(self, x, a):
        if (self.Qapproximation):
            return 1
        elif self.sparse:
            if x in self.Visits:
                return self.Visits[x][a]
            else:
                return 0
        else:
            return self.Visits[x,a]


    def getAlphaVisitsInc(self, x, a):
        s = self.getVisits(x,a)
        try: #TODO debug here
            a = 1.0/float(s)
        except:
            a = 1.0
        #print("visits: %d, a = %.6f" %(s,a))
        return a # math.sqrt(s)

    def getSumVisits(self, x):
        r = 0
        for a in range(0,self.nactions): 
            r += self.getVisits(x,a)
        return r

    def choose_action(self, x):  # choose action from state x

        if (x is None):
            print('ERROR!!! Choose action from invalid state!!!')

        if (self.epsilon < -1):
            maxIter = 100
            s = self.getSumVisits(x)
            p = min(float(s)/maxIter, 1.0)
            epsilon = 0.9 * (1.0 - p) + 0.1
            #print("  -- iter = %d  -- epsilon = %f" %(s,epsilon))
        elif (self.epsilon < 0):
            maxIter = 10000
            s = self.iteration #getSumVisits(x)
            p = min(float(s)/maxIter, 1.0)
            epsilon = 0.9 * (1.0 - p) + 0.1
            #print("  -- iter = %d  -- epsilon = %f" %(s,epsilon))
        else:
            epsilon = self.epsilon
        self.best_action = False
        ar = random.random()
        #if (self.debug):
        #    print(" .. random %f < epsilon = %f" %(ar,epsilon))
        
        if ((not self.optimal) and (not self.option_enabled) and ar<epsilon):
            # Random action
            chosen_a = random.randint(0,self.nactions-1)
            #if (self.debug):
            #    print(" .. random choice ",chosen_a)

        else:
            # Choose the action that maximizes expected reward.            
            self.best_action = True
            Qa = self.getQA(x)
            va = np.argmax(Qa)
            maxs = [i for i,v in enumerate(Qa) if v == Qa[va]]
            #print(" ... Qa = ",Qa,"  va = ",va,"  maxs = ",maxs)
            if len(maxs) > 1:
                #if self.command in maxs:
                #    chosen_a = self.command
                if self.optimal:
                    chosen_a = maxs[0]
                else:
                    chosen_a = random.choice(maxs)
                    #if (self.debug):
                    #    print(" .. action choice among ",maxs," : ",chosen_a)
                    
            else:
                chosen_a = va
            #if (self.debug):
            #    print(" .. best choice ",chosen_a)


        # check state-action failures

        if (x,chosen_a) in self.SA_failure:
            # check non-failure actions for this state
            nfa = []
            for ai in range(0,self.nactions):
                if (x,ai) not in self.SA_failure:
                    nfa.append(ai)
            if (len(nfa)>0):
                chosen_a = random.choice(nfa)            

        return chosen_a

        
    def decision(self, x):
        a = self.choose_action(x)
        if self.debug:
            print("+++ Q [%d] = " %(x), end='')
            self.printQA(self.getQA(x))
            c=' '
            if (self.best_action):
                c='*'
            print(" -  Decision %s %s" %(self.action_names[a],c))
        return a

    # result of execution of action
    def notify(self, x, a, r, x2):

        # DETERMINISTIC / VERY CONSERVATIVE CASE
        # negative reward considered a failure, (x,a) added to SA_failure list
        if (r<0 and (x,a) not in self.SA_failure):
            self.SA_failure.append((x,a))   # new state-action failure
        else:
            # if x2 has all SA failures (x2,a2) for each a2
            # then also (x,a) is a SA failure
            saf = True
            for a2 in range(0,self.nactions):
                saf = saf and (x2,a2) in self.SA_failure
            if saf:
                self.SA_failure.append((x,a))
    
        self.incVisits(x,a)
        self.episode.append((x,a,r))

        if (self.lambdae>0):
            self.setEligibility(x,a)

        if (self.debug):
            print("*** Q update %d with r: %f ***" %(x,r))
        
        if (self.nstepsupdates<1):
            self.updateQ(x,a,r,x2)
        else:
            kn = len(self.episode) - self.nstepsupdates
            if kn>=0:
                self.updateQ_n(kn,x2) # update state-action n-steps back


    def notify_endofepisode(self, iter):
        self.iteration = iter
        if (self.nstepsupdates>0):
            kn = max(0,len(self.episode) - self.nstepsupdates)+1
            while (kn < len(self.episode)):
                self.updateQ_n(kn,None) # update state-action n-steps back
                kn += 1
        self.episode = [] # list of (x,a,r) for this episode
        #print("reset e")
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
        #if (self.debug):
        #    print(" etraces: %d " %(len(self.etraces)))


    def updateEligibility(self, x, a, alpha, delta):
    
        if (self.debug):
            print("  updating e: %d %d ... -  etraces: %d " %(x,a,len(self.etraces)))
        for e in self.etraces:
            # update Q table
            if (delta!=0):
                if (alpha<0):
                    alpha = self.getAlphaVisitsInc(e[0],e[1])
                q = alpha * delta * self.etraces[e]
                self.addQ(e[0],e[1],q)
                if (self.debug):
                    #print("  -- e ",e," ",self.etraces[e])
                    #print("  -- e x:",e[0]," a:",e[1])
                    #print("  -- alpha: ",alpha,"  delta: ", delta)
                    #print("  -- Q(e) = ", self.getQ(e[0],e[1]))
                    print("  -e- Q[%d] = " %(e[0]), end='')
                    self.printQA(self.getQA(e[0]))
                    print('')
        if (self.debug):
            print("++\n")
        # clear traces after update
        #self.etraces = {}

            
    # NOT USED. JUST FOR EXPLANATION
    def updateQ(self,x,a,r,x2):
    
        if (self.optimal):  # executes best policy, no updates
            return

        # Q of current state
        prev_Q = self.getQ(x,a)

        vQa = self.getActionValue(x2)
        
        delta = r + self.gamma * vQa - prev_Q

        if (self.debug):
            print(' == ',x,' A: ',a,' -> r: ',r,' -> ',x2,' prev_Q: ', prev_Q, '  vQa: ', vQa)
            print(' == Q update Q ',x,',',a,' <-  ...  Q ',x2,' = ', vQa)
        
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

        #if (self.debug):
        #    print("updateQ_n ",kn, " optimal = ",self.optimal)
        
        if (self.optimal):  # executing best policy, no updates
            return

        #if (self.debug):
        #    print("debug updateQ_n ... ",kn)

        if (kn<0):  # kn not valid
            return

        ep = self.episode[kn]
        x_kn = ep[0]
        a_kn = ep[1]
        g = self.rreturn(kn, self.nstepsupdates) # n-steps return from state x_{kn}

        #if (abs(g)<0.001): # small values, no updates
        #    return


        if (self.debug):
            print("return_pre = ",g)

        Qx2 = 0
        # if not at the end of the episode
        if (not x2 is None and x_kn!=x2):
            Qx2 = self.getActionValue(x2)
            g += math.pow(self.gamma, self.nstepsupdates) * Qx2 # expected value in next state

        q_kn = self.getQ(x_kn,a_kn)
        delta = (g - q_kn)

        if (self.debug):
            print("debug updateQ_n ")
            print("x_kn = ",x_kn, "  a_kn = ",a_kn, "x2 = ",x2)
            print("Q[%d] = %.3f" %(x2,Qx2))
            print("return_complete = ",g)
            print("Q[%d] = %.3f" %(x_kn,q_kn))
            print("delta = ",delta)

        #if (abs(delta)<0.001):  # time optimization
        #    return

        if (self.lambdae>0):
            self.updateEligibility(x_kn,a_kn,self.alpha,delta)
        else:
            if (self.alpha>=0):
                alpha = self.alpha
            else:
                alpha = self.getAlphaVisitsInc(x_kn,a_kn)
            q = alpha * delta    
            self.addQ(x_kn,a_kn,q)


    def printQA(self, qv):
        print("[ ", end='')
        for a in qv:
            print("%.3f "%a, end='')
        print("]", end='')



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
        
