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


# Reward automa

class RewardAutoma(object):

    def __init__(self, brick_cols=0): # brick_cols=0 -> RA disabled
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
            'RAGoal':1000,       # goal of reward automa
            'RAFail':0,         # fail of reward automa
            'GoodBrick':100,      # good brick removed for next RA state
            'WrongBrick':0      # wrong brick removed for next RA state
        }
        self.goalreached = 0 # number of RA goals reached for statistics
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = 0
        self.last_node = self.current_node


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
        # RA disabled
        if (self.brick_cols==0):
            return 0
            
        reward = 0
        
        for b in self.game.last_brikcsremoved:
            if b.i == self.current_node:
                reward += self.STATES['GoodBrick']
                #print 'Hit right brick for next RA state'
            else:
                reward += self.STATES['WrongBrick']
                #print 'Hit wrong brick for next RA state'

        f = np.zeros(self.brick_cols)
        for c in range(0,self.brick_cols):
            f[c] = self.check_free_cols([c])

        if (self.current_node<self.brick_cols): # 0 ... brick_cols
            if f[self.current_node]:
                self.last_node = self.current_node
                self.current_node += 1
                reward += self.STATES['RAGoal']
                #print("  -- RA state transition to %d, " %(self.current_node))
                if (self.current_node==self.RAGoal):
                    # print("  <<< RA GOAL >>>")
                    reward += self.STATES['RAGoal']
                    self.goalreached += 1
            else:
                for c in range(self.current_node, self.brick_cols):
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

        return reward




class BreakoutSRA(BreakoutS):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test'):
        BreakoutS.__init__(self,brick_rows, brick_cols, trainsessionname)
        self.RA = RewardAutoma(brick_cols)
        self.RA.init(self)
        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':0,
            'PaddleNotMoving':0,
            'Scores':0,    # brick removed
            'Hit':0,        # paddle hit
            'Goal':0,     # level completed
        }


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

    def update(self, a):
        super(BreakoutSRA, self).update(a)
        self.current_reward += self.RA.update()
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
            
        s = 'Iter %6d, b_hit: %3d, p_hit: %3d, na: %4d, r: %5d, RA: %d, mem: %d  %c' %(self.iteration, self.score, self.paddle_hit_count,self.numactions, self.cumreward, RAnode, len(self.agent.Q), ch)

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
            print('-----------------------------------------------------------------------')
            print("%s %6d avg last 100: reward %.1f | RA %.2f | p goals %.1f %% <<<" %(self.trainsessionname, self.iteration, float(self.cumreward100/100), float(self.cumscore100)/100, float(self.RA.goalreached*100)/numiter))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.RA.goalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        self.resfile.write("%d,%d,%d,%d\n" % (RAnode, self.cumreward, self.goal_reached(),self.numactions))
        self.resfile.flush()



class BreakoutNRA(BreakoutN):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test', RAenabled=True):
        BreakoutN.__init__(self,brick_rows, brick_cols, trainsessionname)
        RA_cols = 0
        if (RAenabled):
            RA_cols = brick_cols
        self.RA = RewardAutoma(RA_cols)
        self.RA.init(self)
        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':0,
            'PaddleNotMoving':0,
            'Scores':0,    # brick removed
            'Hit':1,       # paddle hit
            'Goal':0,      # level completed
        }


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

    def update(self, a):
        super(BreakoutNRA, self).update(a)
        self.current_reward += self.RA.update()
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

        if (r<0):
            print("Neg reward: %.1f" %r)
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
            
        s = 'Iter %6d, b_hit: %3d, p_hit: %3d, na: %4d, r: %5d, RA: %d, mem: %d  %c' %(self.iteration, self.score, self.paddle_hit_count,self.numactions, self.cumreward, RAnode, len(self.agent.Q), ch)

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
            print('----------------------------------------------------------------------------------')
            print("%s %6d avg last 100: reward %.1f | RA %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration, float(self.cumreward100/100), float(self.cumscore100)/100, float(self.RA.goalreached*100)/numiter))
            print('----------------------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.RA.goalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        self.resfile.write("%d,%d,%d,%d\n" % (RAnode, self.cumreward, self.goal_reached(),self.numactions))
        self.resfile.flush()







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
  
  
