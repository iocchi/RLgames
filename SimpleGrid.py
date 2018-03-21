import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

black = [0, 0, 0]
white = [255,255,255]
grey = [180,180,180]
orange = [180,100,20]


STATES = {
    'Init':0,
    'Alive':0,
    'Dead':0,
    'NotMoving':0,
    'Score':1000,  
    'Higher':0,  # towards higher values of (x+y)g 
    'Hit':0,        
}



class SimpleGrid(object):

    def __init__(self, rows=3, cols=3, trainsessionname='test'):

        self.agent = None
        self.isAuto = True
        self.gui_visible = False
        self.userquit = False
        self.optimalPolicyUser = False  # optimal policy set by user
        self.trainsessionname = trainsessionname
        self.rows = rows
        self.cols = cols
        
        # Configuration
        self.pause = False # game is paused
        self.debug = False
        
        self.sleeptime = 0.0
        self.command = 0
        self.iteration = 0
        self.cumreward = 0
        self.cumreward100 = 0 # cum reward for statistics
        self.cumscore100 = 0 
        self.ngoalreached = 0
        
        self.hiscore = 0
        self.hireward = -1000000
        self.resfile = open("data/"+self.trainsessionname +".dat","a+")
        self.elapsedtime = 0 # elapsed time for this experiment

        self.win_width = 480
        self.win_height = 520

        self.size_square = 40
        self.offx = 40
        self.offy = 100
        self.radius = 5

        self.action_names = ['--','<-','->','^','v'] # 0: not moving, 1: left, 2: right, 3: up, 4: down        

        if (self.cols>10):
            self.win_width += self.size_square * (self.cols-10)
        if (self.rows>10):
            self.win_height += self.size_square * (self.rows-10)

        pygame.init()

        #allows for holding of key
        #pygame.key.set_repeat(1,0)

        # self.reset()

        self.screen = pygame.display.set_mode([self.win_width,self.win_height])
        self.myfont = pygame.font.SysFont("Arial",  30)
        
        
    def init(self, agent):  # init after creation (uses args set from cli)
        if (not self.gui_visible):
            pygame.display.iconify()

        self.agent = agent
        self.nactions = 5  # 0: not moving, 1: left, 2: right, 3: up, 4: down        
        ns = self.rows * self.cols
        print('Number of states: %d' %ns)
        print('Number of actions: %d' %self.nactions)
        self.agent.init(ns, self.nactions) # 1 for RA not used here
        self.agent.set_action_names(self.action_names)


        
    def reset(self):
        
        self.pos_x = 0
        self.pos_y = 0

        self.score = 0
        self.cumreward = 0
        self.cumscore = 0  
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state

        self.prev_state = None # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.newstate = True # new state reached
        self.numactions = 0 # number of actions in this episode
        self.iteration += 1

        self.agent.optimal = self.optimalPolicyUser or (self.iteration%100)==0 # False #(random.random() < 0.5)  # choose greedy action selection for the entire episode

    def savedata(self):
         return [self.iteration, self.hiscore, self.hireward, self.elapsedtime]
         #, self.RA.visits, self.RA.success]

    def loaddata(self,data):
         self.iteration = data[0]
         self.hiscore = data[1]
         self.hireward = data[2]
         self.elapsedtime = data[3]
         #self.RA.visits = data[4]
         #self.RA.success = data[5]
        
    def getstate(self):
        x = self.pos_x + self.cols * self.pos_y        
        return x


    def goal_reached(self):
        return self.pos_x == self.cols-1 and self.pos_y == self.rows-1  
        
        
    def update(self, a):
        
        self.command = a

        self.prev_state = self.getstate() # remember previous state
        
        # print " == Update start ",self.prev_state," action",self.command 
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        #print('self.current_reward = 0')
        self.numactions += 1 # total number of actions axecuted in this episode
        
        # while (self.prev_state == self.getstate()):
        
        if (self.firstAction):
            self.firstAction = False
            self.current_reward += STATES['Init']
        
        if self.command == 0:  # not moving
            # do nothing
            self.current_reward += STATES['NotMoving']
            pass
        elif self.command == 1:  # moving left
            self.pos_x -= 1
            if (self.pos_x < 0):
                self.pos_x = 0 
                self.current_reward += STATES['Hit']
        elif self.command == 2:  # moving right
            self.pos_x += 1
            if (self.pos_x >= self.cols):
                self.pos_x = self.cols-1
                self.current_reward += STATES['Hit']
        elif self.command == 3:  # moving up
            self.pos_y += 1
            if (self.pos_y >= self.rows):
                self.pos_y = self.rows-1
                self.current_reward += STATES['Hit']
        elif self.command == 4:  # moving down
            self.pos_y -= 1
            if (self.pos_y< 0):
                self.pos_y = 0 
                self.current_reward += STATES['Hit']

        self.current_reward += STATES['Alive']
        
        xy = self.pos_x + self.pos_y
        if (xy > self.score):
            self.score = xy
            self.current_reward += STATES['Higher']
        
        # check if episode terminated
        if self.goal_reached():
            self.current_reward += STATES['Score']
            self.ngoalreached += 1
            self.finished = True
        if (self.numactions>(self.cols+self.rows)*2):
            self.current_reward += STATES['Dead']
            self.finished = True

        #print " ** Update end ",self.getstate(), " prev ",self.prev_state
        


    def input(self):
    
        self.usercommand = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.usercommand = 1
                elif event.key == pygame.K_RIGHT:
                    self.usercommand = 2
                elif event.key == pygame.K_UP:
                    self.usercommand = 3
                elif event.key == pygame.K_DOWN:
                    self.usercommand = 4
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    print("Game paused: %s" %self.pause)
                elif event.key == pygame.K_a:
                    self.isAuto = not self.isAuto
                elif event.key == pygame.K_s:
                    self.sleeptime = 1.0
                elif event.key == pygame.K_d:
                    self.sleeptime = 0.07
                elif event.key == pygame.K_f:
                    self.sleeptime = 0.005
                    self.agent.debug = False
                elif event.key == pygame.K_g:
                    self.sleeptime = 0.0
                    self.agent.debug = False
                elif event.key == pygame.K_o:
                    self.optimalPolicyUser = not self.optimalPolicyUser
                    print("Best policy: %s" %self.optimalPolicyUser)
                elif event.key == pygame.K_q:
                    self.userquit = True
                    print("User quit !!!")
                    
        return True

    def getUserAction(self):
        while (self.usercommand<0 and not self.isAuto):
            self.input()
            time.sleep(0.2)
        if (not self.isAuto):
            self.command = self.usercommand
        return self.command

    def getreward(self):
        r = self.current_reward
        self.cumreward += r
        return r


    def print_report(self, printall=False):
        toprint = printall
        ch = ' '
        if (self.agent.optimal):
            ch = '*'
            toprint = True
      
        s = 'Iter %6d, sc: %3d, na: %4d, r: %5d %c' %(self.iteration, self.score,self.numactions, self.cumreward, ch)

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
        self.cumscore100 += self.score
        numiter = 100
        if (self.iteration%numiter==0):
            #self.doSave()
            pgoal = float(self.ngoalreached*100)/numiter
            print('-----------------------------------------------------------------------')
            print("%s %6d avg last 100: reward %d | score %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration,self.cumreward100/100, float(self.cumscore100)/100, pgoal))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0  
            self.cumscore100 = 0 
            self.ngoalreached = 0

        sys.stdout.flush()
        
        self.resfile.write("%d,%d,%d,%d\n" % (self.score, self.cumreward, self.goal_reached(),self.numactions))
        self.resfile.flush()


    def draw(self):
        self.screen.fill(white)

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (20, 10))

        #count_label = self.myfont.render(str(self.paddle_hit_count), 100, pygame.color.THECOLORS['brown'])
        #self.screen.blit(count_label, (70, 10))

        x = self.getstate()
        cmd = ' '
        if self.command==1:
            cmd = '<'
        elif self.command==2:
            cmd = '>'
        elif self.command==3:
            cmd = '^'
        elif self.command==4:
            cmd = 'v'
        s = '%d %s' %(x,cmd)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))
        

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-200, 10))
        if (self.agent.optimal):
            opt_label = self.myfont.render("Best", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(opt_label, (self.win_width-100, 10))
        
        dx = int(self.offx + self.pos_x * self.size_square)
        dy = int(self.offy + (self.rows-self.pos_y-1) * self.size_square)
        for i in range (0,self.cols+1):
            ox = self.offx + i*self.size_square
            pygame.draw.line(self.screen, black, [ox, self.offy], [ox, self.offy+self.rows*self.size_square])
        for i in range (0,self.rows+1):
            oy = self.offy + i*self.size_square
            pygame.draw.line(self.screen, black, [self.offx , oy], [self.offx + self.cols*self.size_square, oy])

        
        pygame.draw.circle(self.screen, orange, [dx+self.size_square//2, dy+self.size_square//2], 2*self.radius, 0)

        pygame.display.update()



    def quit(self):
        self.resfile.close()
        pygame.quit()

