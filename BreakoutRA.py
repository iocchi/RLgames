# A Q-learning Agent which plays breakout well (won't lose).
# from https://github.com/lincerely/breakout-Q
#
# The breakout game is based on CoderDojoSV/beginner-python's tutorial
#
# Adapted and updated for teaching purposes
# Luca Iocchi 2017


import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

from Breakout import *


STATES = {
    'RAGoal':600,        # goal of reward automa
    'RAFail':-30,        # fail of reward automa
}


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
        
        self.goalreached = 0 # number of RA goals reached for statistics
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = 0
        self.last_node = self.current_node


    # RewardAutoma Transition
    def update(self):
        # RA disabled
        if (self.brick_cols==0):
            return 0
            
        reward = 0
        f = np.zeros(self.brick_cols)
        for c in range(0,self.brick_cols):
            f[c] = self.game.check_free_cols([c])

        if (self.current_node<self.brick_cols): # 0 ... brick_cols
            if f[self.current_node]:
                self.last_node = self.current_node
                self.current_node += 1
                reward += STATES['RAGoal']/self.brick_cols
                #print("  -- RA state transition to %d, " %(self.current_node))
                if (self.current_node==self.RAGoal):
                    # print("  <<< RA GOAL >>>")
                    reward += STATES['RAGoal']
                    self.goalreached += 1
            else:
                for c in range(self.current_node, self.brick_cols):
                    if f[c]:
                        self.last_node = self.current_node
                        self.current_node = self.RAFail  # FAIL
                        reward += STATES['RAFail']
                        #print("  *** RA FAIL *** ")
                        break

        elif (self.current_node==self.RAGoal): #  GOAL
            pass

        elif (self.current_node==self.RAFail): #  FAIL
            pass

        return reward





class BreakoutRA(Breakout):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test'):
        Breakout.__init__(self,brick_rows, brick_cols, trainsessionname)
        self.RA = RewardAutoma(brick_cols)
        self.RA.init(self)

    def setStateActionSpace(self):
        self.n_ball_x = self.win_width/resolutionx+1
        self.n_ball_y = self.win_height/resolutiony+1
        self.n_ball_dir = 10 # ball going up (0-5) or down (6-9)
                        # ball going left (1,2) straight (0) right (3,4)
        self.n_paddle_x = self.win_width/resolutionx+1
        self.nactions = 3  # 0: not moving, 1: left, 2: right
        
        self.nstates = self.n_ball_x * self.n_ball_y * self.n_ball_dir * self.n_paddle_x * self.RA.nRAstates
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)
 

    def getstate(self):
        #diff_paddle_ball = (int(self.ball_x)-self.paddle_x+self.win_width)/resolution
        resx = resolutionx # highest resolution
        resy = resolutiony # highest resolution
        if (self.ball_y<self.win_height/3): # upper part, lower resolution
            resx *= 3
            resy *= 3
        elif (self.ball_y<2*self.win_height/3): # lower part, medium resolution
            resx *= 2
            resy *= 2
        
        ball_x = int(self.ball_x)/resx
        ball_y = int(self.ball_y)/resy
        ball_dir=0
        if self.ball_speed_y > 0: # down
            ball_dir += 5
        if self.ball_speed_x < -2.5: # quick-left
            ball_dir += 1
        elif self.ball_speed_x < 0: # left
            ball_dir += 2
        elif self.ball_speed_x > 2.5: # quick-right
            ball_dir += 3
        elif self.ball_speed_x > 0: # right
            ball_dir += 4

        if self.simple_state:
            paddle_x = 0 
        else:
            paddle_x = int(self.paddle_x)/resx
        
        return ball_x  + self.n_ball_x * ball_y + self.n_ball_y * ball_dir + self.n_ball_dir * paddle_x + self.n_paddle_x * self.RA.current_node

        
    def reset(self):
        super(BreakoutRA, self).reset()
        self.RA.reset()
        
    
    # check if a column is free (used by RA)
    def check_free_cols(self, cols):
        cond = True
        for c in cols:
            for j in range(0,self.brick_rows):
                if (self.bricksgrid[c][j]==1):
                    cond = False
                    break
        return cond


    def update(self, a):
        super(BreakoutRA, self).update(a)
        self.RA.update()
         
    def goal_reached(self):
        return self.RA.current_node==self.RA.RAGoal
        
        
    def getreward(self):
        r = self.current_reward
        if (self.current_reward>0 and self.RA.current_node>0 and self.RA.current_node<=self.RA.RAGoal):
            r *= (self.RA.current_node+1)
            #print "MAXI REWARD ",r
        elif (self.current_reward>0 and self.RA.current_node==self.RA.RAFail):  # FAIL RA state
            r = 0
        self.cumreward += r
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
            
        s = 'Iter %6d, sc: %3d, p_hit: %3d, na: %4d, r: %5d, RA: %d %c' %(self.iteration, self.score, self.paddle_hit_count,self.numactions, self.cumreward, RAnode, ch)

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
        numiter = 100
        if (self.iteration%numiter==0):
            #self.doSave()
            print('-----------------------------------------------------------------------')
            print("%s %6d Avg Reward last 100 runs:  >>> %d <<<  p goals >>> %.1f %% <<<" %(self.trainsessionname, self.iteration,self.cumreward100/100,float(self.RA.goalreached*100)/numiter))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0
            self.RA.goalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
        self.resfile.write("%d,%d,%d\n" % (self.score, self.cumreward, RAnode))
        self.resfile.flush()
