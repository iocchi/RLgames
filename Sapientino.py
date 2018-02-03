import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs

COLORS = ['red', 'green', 'blue', 'pink', 'brown', 'gray', 'purple' ]

TOKENS = [ ['r1', COLORS[0], 0, 0],  ['r2', 'red', 1, 1], ['r3', 'red', 6, 3],   
    ['g1', 'green', 4, 0], ['g2', 'green', 5, 2], ['g3', 'green', 5, 4],
    ['b1', 'blue', 1, 3], ['b2', 'blue', 2, 4],  ['b3', 'blue', 6, 0], 
    ['p1', 'pink', 2, 1], ['p2', 'pink', 2, 3], ['p3', 'pink', 4, 2], 
    ['n1', 'brown', 3, 0], ['n2', 'brown', 3, 4], ['n3', 'brown', 6, 1],
    ['y1', 'gray', 0, 2], ['y2', 'gray', 3, 1], ['y3', 'gray', 4, 3],
    ['u1', 'purple', 0, 4], ['u2', 'purple', 1, 0], ['u3', 'purple', 5, 1]
]

# only positive rewards

STATES = {
    'Init':0,
    'Alive':0,
    'Dead':0,
    'Score':0,
    'Hit':0,
    'GoodColor':0,
    'GoalStep':100,
    'RAFail':0,
    'RAGoal':100
}

# Reward automa

class RewardAutoma(object):

    def __init__(self, ncol, nvisitpercol):
        # RA states
        self.ncolors = ncol
        self.nvisitpercol = nvisitpercol
        self.nRAstates = math.pow(2,self.ncolors*3)+2  # number of RA states
        self.RAGoal = self.nRAstates
        self.RAFail = self.nRAstates+1        
        self.goalreached = 0 # number of RA goals reached for statistics
        self.reset()
        
    def init(self, game):
        self.game = game
        
    def reset(self):
        self.current_node = 0
        self.last_node = self.current_node
        self.past_colors = []
        self.consecutive_turns = 0 # number of consecutive turn actions
        self.countupdates = 0 # count state transitions (for the score)

    def encode_tokenbip(self):
        c = 0
        b = 1
        for t in TOKENS:
            c = c + self.game.tokenbip[t[0]] * b
            b *= 2
        return c

    # RewardAutoma Transition
    def update(self, a=None): # last action executed
        reward = 0
        self.last_node = self.current_node

        # check consecutive turns in differential mode
        if (a == 0 or a == 1): # turn left/right
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 0

        if (self.consecutive_turns>=4):
            self.current_node = self.RAFail  # FAIL
            reward += STATES['RAFail']   

        # check double bip
        for t in self.game.tokenbip:
            if self.game.tokenbip[t]>1:                
                self.current_node = self.RAFail  # FAIL
                reward += STATES['RAFail']
                #print("  *** RA FAIL (two bips) *** ")

        if (self.current_node != self.RAFail):
            self.current_node = self.encode_tokenbip()

            #print("  -- encode tokenbip: %d" %self.current_node)
            # Check rule
            # nvisitpercol
            c = np.zeros(self.ncolors)
            kc = -1
            #print self.game.colorbip
            for i in range(len(COLORS)):
                if (self.game.colorbip[COLORS[i]]>self.nvisitpercol):
                    self.current_node = self.RAFail
                    break
                elif (self.game.colorbip[COLORS[i]]<self.nvisitpercol):
                    break
                kc = i # last color with nvisitsper col satisfied
            #print("%d visits until color %d" %(self.nvisitpercol,kc))

            if (kc==self.ncolors-1): #  GOAL
                self.current_node = self.RAGoal

            # check bips in colors >= kc+2
            if (self.current_node != self.RAFail and self.current_node != self.RAGoal):
                for i in range(kc+2,len(COLORS)):
                    if (self.game.colorbip[COLORS[i]]>0):
                        #print("RA failure for color %r" %i)
                        self.current_node = self.RAFail
                        break

            if (self.last_node != self.current_node):
                #print "  ++ changed state ++"
                if (self.current_node == self.RAFail):
                    reward += STATES['RAFail']
                #elif (self.last_id_colvisited != kc): # new state in which color has been visited right amunt of time
                #    self.last_id_colvisited = kc
                #    reward += STATES['GoalStep']
                else: # new state good for the goal
                    self.countupdates += 1
                    #reward += STATES['GoalStep']
                    reward += self.countupdates * STATES['GoalStep']
                if (self.current_node == self.RAGoal): #  GOAL
                    reward += STATES['RAGoal']
                    #print("RAGoal")

        #print("  -- RA reward %d" %(reward))


        return reward






class Sapientino(object):

    def __init__(self, rows=5, cols=7, trainsessionname='test', ncol=7, nvisitpercol=2):

        self.agent = None
        self.isAuto = True
        self.gui_visible = False
        self.userquit = False
        self.optimalPolicyUser = False  # optimal policy set by user
        self.trainsessionname = trainsessionname
        self.rows = rows
        self.cols = cols
        self.differential = False
        self.colorsensor = False
        
        # Configuration
        self.pause = False # game is paused
        self.debug = False
        
        self.sleeptime = 0.0
        self.command = 0
        self.iteration = 0
        self.cumreward = 0
        self.cumreward100 = 0 # cumulative reward for statistics
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

        self.action_names = ['<-','->','^','v','x']

        if (self.cols>10):
            self.win_width += self.size_square * (self.cols-10)
        if (self.rows>10):
            self.win_height += self.size_square * (self.rows-10)

        pygame.init()
        pygame.display.set_caption('Sapientino')

        self.screen = pygame.display.set_mode([self.win_width,self.win_height])
        self.myfont = pygame.font.SysFont("Arial",  30)
        
        self.ncolors = ncol
        self.nvisitpercol = nvisitpercol
        self.RA = RewardAutoma(self.ncolors, self.nvisitpercol)
        self.RA.init(self)

        
    def init(self, agent):  # init after creation (uses args set from cli)
        if (not self.gui_visible):
            pygame.display.iconify()

        self.agent = agent
        self.nactions = 5  # 0: left, 1: right, 2: up, 3: down, 4: bip
        ns = self.getSizeStateSpace() * self.RA.nRAstates
        print('Number of states: %d' %ns)
        print('Number of actions: %d' %self.nactions)
        self.agent.init(ns, self.nactions) 
        self.agent.set_action_names(self.action_names)


        
    def reset(self):
        
        self.pos_x = 3
        self.pos_y = 2
        self.pos_th = 90

        self.score = 0
        self.cumreward = 0
        self.cumscore = 0  
        self.gamman = 1.0 # cumulative gamma over time
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state

        self.prev_state = 0 # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.newstate = True # new state reached
        self.numactions = 0 # number of actions in this episode
        self.iteration += 1

        self.agent.optimal = self.optimalPolicyUser or (self.iteration%100)==0 # False #(random.random() < 0.5)  # choose greedy action selection for the entire episode
        self.tokenbip = {}
        self.colorbip = {}        
        for t in TOKENS:
            self.tokenbip[t[0]] = 0
            self.colorbip[t[1]] = 0
        self.countbip=0
        self.RA.reset()

        
    def getSizeStateSpace(self):
        self.nstates = self.rows * self.cols
        if (self.differential):
            self.nstates *= 4
        if (self.colorsensor):
            self.nstates *= self.ncolors+1
        return self.nstates


    def getstate(self):
        x = self.pos_x + self.cols * self.pos_y
        if (self.differential):
            x += (self.pos_th/90) * (self.rows * self.cols)
        if (self.colorsensor):
            x += self.encode_color() * (self.rows * self.cols * 4)
        x += self.nstates * self.RA.current_node     
        return x


    def goal_reached(self):
        return self.RA.current_node==self.RA.RAGoal


    def update_color(self):
        self.countbip += 1
        colfound = None
        for t in TOKENS:
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                self.tokenbip[t[0]] += 1 # token id
                self.colorbip[t[1]] += 1 # color
                colfound = t[1]
        #print ("pos %d %d %d - col %r" %(self.pos_x, self.pos_y, self.pos_th, colfound))


    def check_color(self):
        r = ' '
        for t in TOKENS:
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                r = t[1]
                break
        return r

 
    def encode_color(self):
        r = 0
        for t in TOKENS:
            r += 1
            if (self.pos_x == t[2] and self.pos_y == t[3]):
                break
        return r

        
    def update(self, a):
        
        self.command = a

        self.prev_state = self.getstate() # remember previous state
        
        # print " == Update start ",self.prev_state," action",self.command 
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        self.numactions += 1 # total number of actions axecuted in this episode
        
        white_bip = False
        
        if (self.firstAction):
            self.firstAction = False
            self.current_reward += STATES['Init']
        

        if (not self.differential):
            # omni directional motion
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
        else:
            # differential motion
            if self.command == 0: # turn left
                self.pos_th += 90
                if (self.pos_th >= 360):
                    self.pos_th -= 360
                #print ("left") 
            elif self.command == 1:  # turn right
                self.pos_th -= 90
                if (self.pos_th < 0):
                    self.pos_th += 360 
                #print ("right") 
            elif (self.command == 2 or self.command == 3):
                dx = 0
                dy = 0
                if (self.pos_th == 0): # right
                    dx = 1
                elif (self.pos_th == 90): # up
                    dy = 1
                elif (self.pos_th == 180): # left
                    dx = -1
                elif (self.pos_th == 270): # down
                    dy = -1
                if (self.command == 3):  # backward
                    dx = -dx
                    dy = -dy
                    #print ("backward") 
                #else:
                    #print ("forward") 
        
                self.pos_x += dx
                if (self.pos_x >= self.cols):
                    self.pos_x = self.cols-1
                    self.current_reward += STATES['Hit']
                if (self.pos_x < 0):
                    self.pos_x = 0 
                    self.current_reward += STATES['Hit']
                self.pos_y += dy
                if (self.pos_y >= self.rows):
                    self.pos_y = self.rows-1
                    self.current_reward += STATES['Hit']
                if (self.pos_y < 0):
                    self.pos_y = 0 
                    self.current_reward += STATES['Hit']


        #print ("pos %d %d %d" %(self.pos_x, self.pos_y, self.pos_th))

        if self.command == 4:  # bip
            self.update_color()
            if (self.check_color()!=' '):
                pass
                #self.current_reward += STATES['Score']
                #if self.debug:
                #    print "bip on color"
            else:
                white_bip = True


        self.current_reward += STATES['Alive']


        if (self.differential):
            RAr = self.RA.update(a)  # consider also turn actions
        else:
            RAr = self.RA.update()

        self.current_reward += RAr

        # set score
        RAnode = self.RA.current_node
        if (RAnode==self.RA.RAFail):
            RAnode = self.RA.last_node

        self.score = self.RA.countupdates

        
        # check if episode finished
        if self.goal_reached():
            self.current_reward += STATES['Score']
            self.ngoalreached += 1
            self.finished = True
        if (self.numactions>(self.cols*self.rows)*10):
            self.current_reward += STATES['Dead']
            self.finished = True
        if (self.RA.current_node==self.RA.RAFail):
            self.finished = True
        if (white_bip):
            self.current_reward += STATES['Dead']
            self.finished = True

        #print " ** Update end ",self.getstate(), " prev ",self.prev_state

        #if (self.finished):
        #    print("  -- final reward %d" %(self.cumreward))            




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
      
        s = 'Iter %6d, sc: %3d, na: %4d, r: %8.2f, mem: %d %c' %(self.iteration, self.score,self.numactions, self.cumreward, len(self.agent.Q), ch)

        if self.score > self.hiscore:
            if self.agent.optimal:
                self.hiscore = self.score
            s += ' HISCORE '
            toprint = True
        if (self.cumreward > self.hireward):
            if self.agent.optimal:
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
            print("%s %6d/%4d avg last 100: reward %.2f | score %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100)/100, float(self.cumscore100)/100, pgoal))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0  
            self.cumscore100 = 0 
            self.ngoalreached = 0

        sys.stdout.flush()
        
        self.resfile.write("%d,%d,%d,%d\n" % (self.score, self.cumreward, self.goal_reached(),self.numactions))
        self.resfile.flush()


    def draw(self):
        self.screen.fill(pygame.color.THECOLORS['white'])

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['black'])
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
        #s = '%d %s %d' %(self.prev_state,cmd,x)
        s = '%s' %(cmd,)
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
            if (self.tokenbip[tk]==1):
                pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], (dx+15,dy+15,self.size_square-30,self.size_square-30))

        # agent position
        dx = int(self.offx + self.pos_x * self.size_square)
        dy = int(self.offy + (self.rows-self.pos_y-1) * self.size_square)
        pygame.draw.circle(self.screen, pygame.color.THECOLORS['orange'], [dx+self.size_square/2, dy+self.size_square/2], 2*self.radius, 0)

        # agent orientation

        ox = 0
        oy = 0
        if (self.pos_th == 0): # right
            ox = self.radius
        elif (self.pos_th == 90): # up
            oy = -self.radius
        elif (self.pos_th == 180): # left
            ox = -self.radius
        elif (self.pos_th == 270): # down
            oy = self.radius

        pygame.draw.circle(self.screen, pygame.color.THECOLORS['black'], [dx+self.size_square/2+ox, dy+self.size_square/2+oy], 5, 0)

        pygame.display.update()


    def quit(self):
        self.resfile.close()
        pygame.quit()
        

        

        
class SapientinoExt(Sapientino):

    def __init__(self, rows=5, cols=7, trainsessionname='test', ncol=7, nvisitpercol=2):
        Sapientino.__init__(self, rows, cols, trainsessionname, ncol, nvisitpercol)
        self.ncol = ncol
        
    def getSizeStateSpace(self):
        self.origns = super(SapientinoExt, self).getSizeStateSpace()
        # all color status
        self.bip_ns = 2
        self.col_ns = self.ncol + 1
        ns = self.origns * self.bip_ns * self.col_ns
        return ns

    def currentcolor(self):
        scol = self.check_color()
        r = self.ncol
        i = 0
        while (i<self.ncol*3):
            if TOKENS[i][1]==scol:
                r=i
                break
            i += 3
        return r/3


    def getstate(self):
        x = super(SapientinoExt, self).getstate()
        f = 1
        if self.command == 4:
            bx = 1
        else:
            bx = 0
        cx = self.currentcolor()
        #print '  extended state bx %d cx %d ' %(bx,cx)
        x = x + self.origns * bx + (self.origns * self.bip_ns) * cx
        return x


        
class SapientinoExt2(Sapientino):

    def __init__(self, rows=5, cols=7, trainsessionname='test', ncol=7, nvisitpercol=2):
        Sapientino.__init__(self, rows, cols, trainsessionname, ncol, nvisitpercol)
        self.ncol = ncol
        
    def getSizeStateSpace(self):
        self.origns = super(SapientinoExt2, self).getSizeStateSpace()
        # all color status
        col_ns = pow(8,self.ncol)
        ns = self.origns * col_ns
        return ns

    def getstate(self):
        x = super(SapientinoExt2, self).getstate()
        f = 1
        tx = 0
        for i in range(0,self.ncol):
            t = TOKENS[i]
            tx += f * self.tokenbip[t[0]]
            f *= 2
        x = x + self.origns * tx
        return x

