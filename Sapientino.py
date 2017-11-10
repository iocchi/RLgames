import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

COLORS = {
    'black':  [0, 0, 0],
    'white':  [255,255,255],
    'grey':   [180,180,180],
    'orange': [180,100,20],
    'red':    [200,0,0],
    'green':  [0,200,0],
    'blue':  [0,0,200]
}



TOKENS = [ ['r1', 'red', 0, 1],  ['r2', 'red', 2, 2],  
    ['g1', 'green', 1, 3], ['g2', 'green', 5, 1],
    ['b1', 'blue', 1, 0], ['b2', 'blue', 4, 2] 
]


STATES = {
    'Init':0,
    'Alive':0,
    'Dead':-100,
    'Score':0,    
    'Hit':-10,
    'GoodColor':1,    
    'RAFail':-100,
    'RAGoal':1000
}

# Reward automa

class RewardAutoma(object):

    def __init__(self):
        # RA states
        self.ncolors = 3
        self.nRAstates = 2*self.ncolors+2  # number of RA states
        self.RAGoal = self.nRAstates-2
        self.RAFail = self.nRAstates-1        
        self.goalreached = 0 # number of RA goals reached for statistics
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = 0
        self.last_node = self.current_node

    def countbipcol(self,col):
        r = 0
        for t in self.tokenbip:

        if (col=='R' and self.game.pred1_bip==1):
            r += 1
        if (col=='R' and self.game.pred2_bip==1):
            r += 1
        if (col=='G' and self.game.pgreen1_bip==1):
            r += 1
        if (col=='G' and self.game.pgreen2_bip==1):
            r += 1
        if (col=='B' and self.game.pblue1_bip==1):
            r += 1
        if (col=='B' and self.game.pblue2_bip==1):
            r += 1
        return r

    def sumbipcol(self,col):
        r = 0
        for t in self.tokenbip:
            
        if (col=='R'):
            r = self.game.pred1_bip + self.game.pred2_bip
        if (col=='G'):
            r = self.game.pgreen1_bip + self.game.pgreen2_bip
        if (col=='B'):
            r = self.game.pblue1_bip + self.game.pblue2_bip
        return r

    # RewardAutoma Transition
    def update(self):
        reward = 0        
        if (self.current_node==0): # todo RED 1
            if (self.game.check_color()=='R'):
                reward += STATES['GoodColor']
            if (self.countbipcol('R')==1):
                reward += STATES['RAGoal']/(self.ncolors*2)
                self.current_node += 1
            elif (self.countbipcol('G')+self.countbipcol('B')>0):
                self.last_node = self.current_node
                self.current_node = self.RAFail  # FAIL
                reward += STATES['RAFail']
                # print("  *** RA FAIL *** ")

        if (self.current_node==1): # todo RED 2
            if (self.game.check_color()=='R'):
                reward += STATES['GoodColor']             
            if (self.countbipcol('R')==2):
                self.last_node = self.current_node
                self.current_node += 1
                reward += STATES['RAGoal']/self.ncolors
                # print("  -- RA state transition to %d, " %(self.current_node))
            elif (self.countbipcol('G')+self.countbipcol('B')>0 or self.sumbipcol('R')>1):
                self.last_node = self.current_node
                self.current_node = self.RAFail  # FAIL
                reward += STATES['RAFail']
                # print("  *** RA FAIL *** ")
                
        elif (self.current_node==2): # todo GREEN 1
            if (self.game.check_color()=='G'):
                reward += STATES['GoodColor']
            if (self.countbipcol('G')==1):
                reward += STATES['RAGoal']/(self.ncolors*2)
                self.current_node += 1
            elif (self.countbipcol('B')>0):
                self.last_node = self.current_node
                self.current_node = self.RAFail  # FAIL
                reward += STATES['RAFail']
                # print("  *** RA FAIL *** ")

        elif (self.current_node==3): # todo GREEN 2
            if (self.game.check_color()=='G'):
                reward += STATES['GoodColor']
            if (self.countbipcol('G')==2):
                self.last_node = self.current_node
                self.current_node += 1
                reward += STATES['RAGoal']/self.ncolors
                # print("  -- RA state transition to %d, " %(self.current_node))
            elif (self.countbipcol('B')>0 or self.sumbipcol('G')>1):
                self.last_node = self.current_node
                self.current_node = self.RAFail  # FAIL
                reward += STATES['RAFail']
                # print("  *** RA FAIL *** ")

        elif (self.current_node==4): # todo BLUE 1
            if (self.game.check_color()=='B'):
                reward += STATES['GoodColor']
            if (self.countbipcol('B')==1):
                reward += STATES['RAGoal']/(self.ncolors*2)   
                self.current_node += 1
             
        elif (self.current_node==5): # todo BLUE 2
            if (self.count('B')==2):
                self.last_node = self.current_node
                self.current_node += 1
                reward += STATES['RAGoal']/self.ncolors
                #print("  -- RA state transition to %d, " %(self.current_node))
                if (self.current_node==self.RAGoal):
                    #print("  <<< RA GOAL >>>")
                    reward += STATES['RAGoal']
                    self.goalreached += 1
            elif (self.sumbipcol('B')>1):
                self.last_node = self.current_node
                self.current_node = self.RAFail  # FAIL
                reward += STATES['RAFail']
                # print("  *** RA FAIL *** ")


        elif (self.current_node==self.RAGoal): #  GOAL
            pass

        elif (self.current_node==self.RAFail): #  FAIL
            pass

        return reward



class Sapientino(object):

    def __init__(self, rows=4, cols=6, trainsessionname='test'):

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

        self.win_width = 480
        self.win_height = 520

        self.size_square = 40
        self.offx = 40
        self.offy = 100
        self.radius = 5

        # color tokens
        #self.pred1 = (0, 1) # row, col
        #self.pred2 = (2, 2)
        #self.pgreen1 = (1, 3) # row, col
        #self.pgreen2 = (5, 1)
        #self.pblue1 = (1, 0) # row, col
        #self.pblue2 = (4, 2)
 


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
        
        self.RA = RewardAutoma()
        self.RA.init(self)

        
    def init(self, agent):  # init after creation (uses args set from cli)
        if (not self.gui_visible):
            pygame.display.iconify()

        self.agent = agent
        self.nactions = 5  # 0: left, 1: right, 2: up, 3: down, 4: bip
        ns = self.rows * self.cols * self.RA.nRAstates
        print('Number of states: %d' %ns)
        self.agent.init(ns, self.nactions) 


        
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

        self.tokenbip = []
        
        self.RA.reset()


        
    def getstate(self):
        x = self.pos_x + self.cols * self.pos_y + (self.cols * self.rows) * self.RA.current_node     
        return x


    def goal_reached(self):
        #return self.score == 6
        return self.RA.current_node==self.RA.RAGoal


    def update_color(self):
        r = ' '
        for t in TOKENS:
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                self.tokenbip.append(t[0])
                r = t[1]
                break
        else:
            return r

    def check_color(self):
        r = ' '
        for t in TOKENS:
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                r = t[1]
                break
        else:
            return r
 
        
    def update(self, a):
        
        self.command = a

        self.prev_state = self.getstate() # remember previous state
        
        # print " == Update start ",self.prev_state," action",self.command 
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        #print('self.current_reward = 0')
        self.numactions += 1 # total number of actions axecuted in this episode
        
        white_bip = False
        
        if (self.firstAction):
            self.firstAction = False
            self.current_reward += STATES['Init']
        
        if self.command == 0: # moving left
            self.pos_x -= 1
            if (self.pos_x < 0):
                self.pos_x = 0 
                self.current_reward += STATES['Hit']
        elif self.command == 1:  # moving right
            self.pos_x += 1
            if (self.pos_x >= self.cols):
                self.pos_x = self.cols-1
                self.current_reward += STATES['Hit']
        elif self.command == 2:  # moving up
            self.pos_y += 1
            if (self.pos_y >= self.rows):
                self.pos_y = self.rows-1
                self.current_reward += STATES['Hit']
        elif self.command == 3:  # moving down
            self.pos_y -= 1
            if (self.pos_y< 0):
                self.pos_y = 0 
                self.current_reward += STATES['Hit']
        elif self.command == 4:  # bip
            if (self.update_color()!=' '):
                self.current_reward += STATES['Score']
                if self.debug:
                    print "*** Score: ",self.score
            else:
                white_bip = True


        self.current_reward += STATES['Alive']

        self.current_reward += self.RA.update()
        
        # check if episode terminated
        if self.goal_reached():
            self.current_reward += STATES['Score']
            self.ngoalreached += 1
            self.finished = True
        if (self.numactions>(self.cols+self.rows)*10):
            self.current_reward += STATES['Dead']
            self.finished = True
        if (self.RA.current_node==self.RA.RAFail):
            self.finished = True
        if (white_bip):
            self.current_reward += STATES['Dead']
            self.finished = True

        #print " ** Update end ",self.getstate(), " prev ",self.prev_state

        


    def input(self):

        self.usercommand = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                isPressed = True
                if event.key == pygame.K_LEFT:
                    self.usercommand = 0
                elif event.key == pygame.K_RIGHT:
                    self.usercommand = 1
                elif event.key == pygame.K_UP:
                    self.usercommand = 2
                elif event.key == pygame.K_DOWN:
                    self.usercommand = 3
                elif event.key == pygame.K_b: # bip
                    self.usercommand = 4
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    print "Game paused: ",self.pause
                elif event.key == pygame.K_a:
                    self.isAuto = not self.isAuto
                elif event.key == pygame.K_s:
                    self.sleeptime = 1.0
                    #self.agent.debug = False
                elif event.key == pygame.K_d:
                    self.sleeptime = 0.07
                    #self.agent.debug = False
                elif event.key == pygame.K_f:
                    self.sleeptime = 0.005
                    #self.agent.debug = False
                elif event.key == pygame.K_g:
                    self.sleeptime = 0.0
                    #self.agent.debug = False
                elif event.key == pygame.K_o:
                    self.optimalPolicyUser = not self.optimalPolicyUser
                    print "Best policy: ",self.optimalPolicyUser
                elif event.key == pygame.K_q:
                    self.userquit = True
                    print "User quit !!!"

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
        
        if (self.current_reward>0 and self.RA.current_node>0 and self.RA.current_node<=self.RA.RAGoal):
            r *= (self.RA.current_node+1)
            # print "MAXI REWARD ",r
        elif (self.current_reward>0 and self.RA.current_node==self.RA.RAFail):  # FAIL RA state
            r = 0
        self.cumreward += r
        return r



        r = self.current_reward
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
        self.score = RAnode


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
        self.screen.fill(pygame.color.THECOLORS['white'])

        score_label = self.myfont.render(str(self.RA.current_node), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (20, 10))

        #count_label = self.myfont.render(str(self.paddle_hit_count), 100, pygame.color.THECOLORS['brown'])
        #self.screen.blit(count_label, (70, 10))

        x = self.getstate()
        cmd = ' '
        if self.command==0:
            cmd = '<'
        elif self.command==1:
            cmd = '>'
        elif self.command==2:
            cmd = '^'
        elif self.command==3:
            cmd = 'v'
        elif self.command==4:
            cmd = 'x'
        s = '%d %s' %(x,cmd)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))
        

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-200, 10))
        if (self.agent.optimal):
            opt_label = self.myfont.render("Best", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(opt_label, (self.win_width-100, 10))


        
        # grid
        for i in range (0,self.cols+1):
            ox = self.offx + i*self.size_square
            pygame.draw.line(self.screen, pygame.color.THECOLORS['black'], [ox, self.offy], [ox, self.offy+self.rows*self.size_square])
        for i in range (0,self.rows+1):
            oy = self.offy + i*self.size_square
            pygame.draw.line(self.screen, pygame.color.THECOLORS['black'], [self.offx , oy], [self.offx + self.cols*self.size_square, oy])


        # color tokens
        for t in TOKENS:
            tk = t[0]
            col = t[1]
            u = t[2]
            v = t[3]
            dx = int(self.offx + u * self.size_square)
            dy = int(self.offy + (self.rows-v-1) * self.size_square)
            sqsz = (dx+5,dy+5,self.size_square-10,self.size_square-10)
            pygame.draw.rect(self.screen, pygame.color.THECOLORS[col], sqsz)
            if tk in self.tokenbip:
                pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], (dx+15,dy+15,self.size_square-30,self.size_square-30))


        # agent position
        dx = int(self.offx + self.pos_x * self.size_square)
        dy = int(self.offy + (self.rows-self.pos_y-1) * self.size_square)
        pygame.draw.circle(self.screen, pygame.color.THECOLORS['orange'], [dx+self.size_square/2, dy+self.size_square/2], 2*self.radius, 0)

        pygame.display.update()



    def quit(self):
        self.resfile.close()
        pygame.quit()

