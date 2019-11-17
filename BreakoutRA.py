#
# Breakout game with reward automa for non-Markovian rewards
#
# Luca Iocchi 2017
#

import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

from Breakout import *

np.set_printoptions(precision=3)

# Reward automa

class RewardAutoma(object):

    def __init__(self, brick_cols=0, left_right=True): # brick_cols=0 -> RA disabled
        # RA states
        self.brick_cols = brick_cols
        if (self.brick_cols>0):
            self.nRAstates = brick_cols+2  # number of RA states
            self.RAGoal = self.nRAstates-2
            self.RAFail = self.nRAstates-1
        else: # RA disabled
            self.nRAstates = 2  # number of RA states
            self.RAGoal = 1 # never reached
            self.RAFail = 2 # never reached

        self.STATES = {
            'RAGoalStep':100,   # goal step of reward automa
            'RAGoal':1000,      # goal of reward automa
            'RAFail':0,         # fail of reward automa
            'GoodBrick':0,      # good brick removed for next RA state
            'WrongBrick':0      # wrong brick removed for next RA state
        }

        self.left_right = left_right
        self.goalreached = 0 # number of RA goals reached for statistics
        self.visits = {} # number of visits for each state
        self.success = {} # number of good transitions for each state
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = 0
        self.last_node = self.current_node
        self.countupdates = 0 # count state transitions (for the score)
        if (self.current_node in self.visits):
            self.visits[self.current_node] += 1
        else:
            self.visits[self.current_node] = 1


    # check if a column is free (used by RA)
    def check_free_cols(self, cols):
        cond = True
        for c in cols:
            for j in range(0,self.game.brick_rows):
                if (self.game.bricksgrid[c][j]==1):
                    cond = False
                    break
        return cond

    # RewardAutoma Transition
    def update(self):
        reward = 0
        state_changed = False

        # RA disabled
        if (self.brick_cols==0):
            return (reward, state_changed)
            
        for b in self.game.last_brikcsremoved:
            if b.i == self.current_node:
                reward += self.STATES['GoodBrick']
                #print 'Hit right brick for next RA state'
            else:
                reward += self.STATES['WrongBrick']
                #print 'Hit wrong brick for next RA state'

        f = np.zeros(self.brick_cols)
        for c in range(0,self.brick_cols):
            f[c] = self.check_free_cols([c])  # vector of free columns

        if (self.current_node<self.brick_cols): # 0 ... brick_cols
            if self.left_right:
                goal_column = self.current_node
                cbegin = goal_column + 1
                cend = self.brick_cols
                cinc = 1
            else:
                goal_column = self.brick_cols - self.current_node - 1
                cbegin = goal_column - 1
                cend = -1
                cinc = -1

            if f[goal_column]:
                state_changed = True
                self.countupdates += 1
                self.last_node = self.current_node
                self.current_node += 1
                reward += self.STATES['RAGoal'] * self.countupdates / self.brick_cols
                #print("  -- RA state transition to %d, " %(self.current_node))
                if (self.current_node==self.RAGoal):
                    # print("  <<< RA GOAL >>>")
                    reward += self.STATES['RAGoal']
                    self.goalreached += 1
            else:
                for c in range(cbegin, cend, cinc):
                    if f[c]:
                        self.last_node = self.current_node
                        self.current_node = self.RAFail  # FAIL
                        reward += self.STATES['RAFail']
                        #print("  *** RA FAIL *** ")
                        break

        elif (self.current_node==self.RAGoal): #  GOAL
            pass

        elif (self.current_node==self.RAFail): #  FAIL
            pass

        if (state_changed):
            if (self.current_node in self.visits):
                self.visits[self.current_node] += 1
            else:
                self.visits[self.current_node] = 1

            if (self.current_node != self.RAFail):
                #print "Success for last_node ",self.last_node
                if (self.last_node in self.success):
                    self.success[self.last_node] += 1
                else:
                    self.success[self.last_node] = 1


        return (reward, state_changed)

    def current_successrate(self):
        s = 0.0
        v = 1.0
        if (self.current_node in self.success):
            s = float(self.success[self.current_node])
        if (self.current_node in self.visits):
            v = float(self.visits[self.current_node])
        #print "   -- success rate: ",s," / ",v
        return s/v

    def print_successrate(self):
        r = []
        for i in range(len(self.success)):
            r.append(float(self.success[i])/self.visits[i])
        print('RA success: %s' %str(r))



class BreakoutSRA(BreakoutS):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test'):
        BreakoutS.__init__(self,brick_rows, brick_cols, trainsessionname)
        self.RA = RewardAutoma(brick_cols)
        self.RA.init(self)
        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':-1,
            'PaddleNotMoving':0,
            'Scores':0,    # brick removed
            'Hit':0,       # paddle hit
            'Goal':0,      # level completed
        }
        self.RA_exploration_enabled = False  # Use options to speed-up learning process
        self.report_str = ''


    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime, self.RA.visits, self.RA.success, self.agent.SA_failure, random.getstate(),np.random.get_state()]

         
    def loaddata(self,data):
        self.iteration = data[0]
        self.hiscore = data[1]
        self.hireward = data[2]
        self.elapsedtime = data[3]
        self.RA.visits = data[4]
        self.RA.success = data[5]
        if (len(data)>6):
            self.agent.SA_failure = data[6]
        if (len(data)>7):
            print('Set random generator state from file.')
            random.setstate(data[7])
            np.random.set_state(data[8])       


    def setStateActionSpace(self):
        super(BreakoutSRA, self).setStateActionSpace()
        self.nstates *= self.RA.nRAstates
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

    def getstate(self):
        x = super(BreakoutSRA, self).getstate()
        return x + (self.n_diff_paddle_ball) * self.RA.current_node

    def reset(self):
        super(BreakoutSRA, self).reset()
        self.RA.reset()
        self.RA_exploration()

    def update(self, a):
        super(BreakoutSRA, self).update(a)
        (RAr, state_changed) = self.RA.update()
        if (state_changed):
            self.RA_exploration()
        self.current_reward += RAr
        if (self.RA.current_node==self.RA.RAFail):
            self.finished = True
         
    def goal_reached(self):
        return self.RA.current_node==self.RA.RAGoal
       
    def getreward(self):
        r = self.current_reward
        #for b in self.last_brikcsremoved:
        #    if b.i == self.RA.current_node:
        #         r += self.STATES['GoodBrick']
                #print 'Hit right brick for next RA state'
        #if (self.current_reward>0 and self.RA.current_node>0 and self.RA.current_node<=self.RA.RAGoal):
            #r *= (self.RA.current_node+1)
            #print "MAXI REWARD ",r

        if (self.current_reward>0 and self.RA.current_node==self.RA.RAFail):  # FAIL RA state
            r = 0
        self.cumreward += self.gamman * r
        self.gamman *= self.agent.gamma
        return r

    def print_report(self, printall=False):
        toprint = printall
        ch = ' '
        if (self.agent.optimal):
            ch = '*'
            toprint = True
            
        RAnode = self.RA.current_node
        if (RAnode==self.RA.RAFail):
            RAnode = self.RA.last_node
            
        s = 'Iter %6d, b_hit: %3d, p_hit: %3d, na: %4d, r: %5d, RA: %d, mem: %d/%d  %c' %(self.iteration, self.score, self.paddle_hit_count,self.numactions, self.cumreward, RAnode, len(self.agent.Q), len(self.agent.SA_failure), ch)

        if self.score > self.hiscore:
            self.hiscore = self.score
            s += ' HISCORE '
            toprint = True
        if self.cumreward > self.hireward:
            self.hireward = self.cumreward
            s += ' HIREWARD '
            toprint = True

        if (toprint):
            print(s)

        self.cumreward100 += self.cumreward
        self.cumscore100 += RAnode
        numiter = 100
        if (self.iteration%numiter==0):
            #self.doSave()
            self.report_str = "%s %6d/%4d avg last 100: reward %.1f | RA %.2f | p goals %.1f %% <<<" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100/100), float(self.cumscore100)/100, float(self.RA.goalreached*100)/numiter)
            print('-----------------------------------------------------------------------')
            print(self.report_str)
            self.RA.print_successrate()
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.RA.goalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        #self.resfile.write("%d,%d,%d,%d,%d\n" % (RAnode, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.write("%d,%d,%d,%d,%d,%d,%d\n" % (self.iteration, self.elapsedtime, RAnode, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.flush()


    def RA_exploration(self):
        if not self.RA_exploration_enabled:
            return
        #print("RA state: ",self.RA.current_node)
        success_rate = max(min(self.RA.current_successrate(),0.9),0.1)
        #print("RA exploration policy: current state success rate ",success_rate)
        er = random.random()
        self.agent.option_enabled = (er<success_rate)
        #print("RA exploration policy: optimal ",self.agent.partialoptimal, "\n")





class BreakoutNRA(BreakoutN):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test', RAenabled=True):
        BreakoutN.__init__(self,brick_rows, brick_cols, trainsessionname)
        RA_cols = 0
        if (RAenabled):
            RA_cols = brick_cols
        self.RA_exploration_enabled = False  # Use options to speed-up learning process

        self.RA = RewardAutoma(RA_cols)
        self.RA.init(self)
        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':-1,
            'PaddleNotMoving':0,
            'Scores':0,    # brick removed
            'Hit':0,       # paddle hit
            'Goal':0,      # level completed
        }


    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime, self.RA.visits, self.RA.success, self.agent.SA_failure, random.getstate(),np.random.get_state()]

         
    def loaddata(self,data):
        self.iteration = data[0]
        self.hiscore = data[1]
        self.hireward = data[2]
        self.elapsedtime = data[3]
        self.RA.visits = data[4]
        self.RA.success = data[5]
        if (len(data)>6):
            self.agent.SA_failure = data[6]
        if (len(data)>7):
            print('Set random generator state from file.')
            random.setstate(data[7])
            np.random.set_state(data[8])   


    def setStateActionSpace(self):
        super(BreakoutNRA, self).setStateActionSpace()
        self.nstates *= self.RA.nRAstates
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

    def getstate(self):
        x = super(BreakoutNRA, self).getstate()
        return x + (self.n_ball_x*self.n_ball_y*self.n_ball_dir*self.n_paddle_x) * self.RA.current_node

    def reset(self):
        super(BreakoutNRA, self).reset()
        self.RA.reset()
        self.RA_exploration()

    def update(self, a):
        super(BreakoutNRA, self).update(a)
        (RAr, state_changed) = self.RA.update()
        if (state_changed):
            self.RA_exploration()
        self.current_reward += RAr
        if (self.RA.current_node==self.RA.RAFail):
            self.finished = True
         
    def goal_reached(self):
        return self.RA.current_node==self.RA.RAGoal
       
    def getreward(self):
        r = self.current_reward
        #if (self.current_reward>0 and self.RA.current_node>0 and self.RA.current_node<=self.RA.RAGoal):
        #    r *= (self.RA.current_node+1)
            #print "MAXI REWARD ",r
        if (self.current_reward>0 and self.RA.current_node==self.RA.RAFail):  # FAIL RA state
            r = 0
        self.cumreward += self.gamman * r
        self.gamman *= self.agent.gamma

        #if (r<0):
        #    print("Neg reward: %.1f" %r)
        return r



    def print_report(self, printall=False):
        toprint = printall
        ch = ' '
        if (self.agent.optimal):
            ch = '*'
            toprint = True
            
        RAnode = self.RA.current_node
        if (RAnode==self.RA.RAFail):
            RAnode = self.RA.last_node
            
        s = 'Iter %6d, b_hit: %3d, p_hit: %3d, na: %4d, r: %5d, RA: %d, mem: %d/%d  %c' %(self.iteration, self.score, self.paddle_hit_count,self.numactions, self.cumreward, RAnode, len(self.agent.Q), len(self.agent.SA_failure), ch)

        if self.score > self.hiscore:
            self.hiscore = self.score
            s += ' HISCORE '
            toprint = True
        if self.cumreward > self.hireward:
            if self.agent.optimal:
                self.hireward = self.cumreward
            s += ' HIREWARD '
            toprint = True

        if (toprint):
            print(s)

        self.cumreward100 += self.cumreward
        self.cumscore100 += RAnode
        numiter = 100
        if (self.iteration%numiter==0):
            #self.doSave()
            print('----------------------------------------------------------------------------------')
            print("%s %6d/%4d avg last 100: reward %.1f | RA %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100/100), float(self.cumscore100)/100, float(self.RA.goalreached*100)/numiter))
            self.RA.print_successrate()
            print('----------------------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.RA.goalreached = 0

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        #self.resfile.write("%d,%d,%d,%d,%d\n" % (RAnode, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.write("%d,%d,%d,%d,%d,%d,%d\n" % (self.iteration, self.elapsedtime, RAnode, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.flush()

    def RA_exploration(self):
        if not self.RA_exploration_enabled:
            return
        #print("RA state: ",self.RA.current_node)
        success_rate = max(min(self.RA.current_successrate(),0.9),0.1)
        #print("RA exploration policy: current state success rate ",success_rate)
        er = random.random()
        self.agent.option_enabled = (er<success_rate)
        #print("RA exploration policy: optimal ",self.agent.partialoptimal, "\n")






class BreakoutSRAExt(BreakoutSRA):

    def setStateActionSpace(self):
        super(BreakoutSRAExt, self).setStateActionSpace()
        self.nstates *= math.pow(2,self.brick_rows*self.brick_cols)
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

    def getstate(self):
        x = super(BreakoutSRAExt, self).getstate()
        return x + (self.n_diff_paddle_ball * self.RA.nRAstates) * self.encodebricks(self.bricksgrid)
        
    def encodebricks(self,brickgrid):
        b = 1
        r = 0
        for i in range(0,self.brick_cols):
            for j in range(0,self.brick_rows):
                if self.bricksgrid[i][j]==1:
                    r += b
                b *= 2
        return r



        
class BreakoutNRAExt(BreakoutNRA):

    def setStateActionSpace(self):
        super(BreakoutNRAExt, self).setStateActionSpace()
        self.nstates *= math.pow(2,self.brick_rows*self.brick_cols)
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

    def getstate(self):
        x = super(BreakoutNRAExt, self).getstate()
        return x + (self.n_ball_x * self.n_ball_y * self.n_ball_dir * self.n_paddle_x * self.RA.nRAstates) * self.encodebricks(self.bricksgrid)
        
    def encodebricks(self,brickgrid):
        b = 1
        r = 0
        for i in range(0,self.brick_cols):
            for j in range(0,self.brick_rows):
                if self.bricksgrid[i][j]==1:
                    r += b
                b *= 2
        return r
  
  
