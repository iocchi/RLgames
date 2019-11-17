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


black = [0, 0, 0]
white = [255,255,255]
grey = [180,180,180]
orange = [180,100,20]
red = [180,0,0]

# game's constant variables
ball_radius = 10
paddle_width = 80
paddle_height = 10

block_width = 60
block_height = 12
block_xdistance = 20
            
resolutionx = 20
resolutiony = 10



class Brick(object):

    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.x = (block_width+block_xdistance)*i+block_xdistance
        self.y = 70+(block_height+8)*j
        self.rect = pygame.Rect(self.x, self.y, block_width, block_height)



class Breakout(object):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test'):

        self.agent = None
        self.RA = None
        self.isAuto = True
        self.gui_visible = False
        self.sound_enabled = True
        self.fire_enabled = False
        self.userquit = False
        self.optimalPolicyUser = False  # optimal policy set by user
        self.evalBestPolicy = False
        self.trainsessionname = trainsessionname
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols

        if (self.brick_cols<5):
            self.block_xdistance = 50

        self.STATES = {
            'Init':0,
            'Alive':0,
            'Dead':0,
            'PaddleNotMoving':0,
            'Scores':10,    # brick removed
            'Hit':0,        # paddle hit
            'Goal':100,     # level completed
        }
        
        # Configuration
        self.deterministic = True   # deterministic ball bouncing
        self.simple_state = False   # simple = do not consider paddle x
        self.paddle_normal_bump = True  # only left/right bounces
        self.paddle_complex_bump = False  # straigth/left/right complex bounces
        self.pause = False # game is paused
        self.debug = False
        
        self.sleeptime = 0.0
        self.init_ball_speed_x = 2
        self.init_ball_speed_y = 5
        self.accy = 1.00
        self.score = 0
        self.ball_hit_count = 0
        self.brick_hit_count = 0
        self.paddle_hit_count = 0
        self.command = 0
        self.iteration = 0
        self.cumreward = 0
        self.cumscore100 = 0 # cumulative score for statistics
        self.cumreward100 = 0 # cumulative reward for statistics
        self.ngoalreached = 0 # number of goals reached for stats
        self.numactions = 0 # number of actions in this run

        self.action_names = ['--','<-','->','o'] # stay, left, right, fire

        # firing variables
        self.fire_posx = 0
        self.fire_posy = 0
        self.fire_speedy = 0 # 0 = not firing, <0 firing up
        
        self.hiscore = 0
        self.hireward = -1000000
        self.vscores = []
        self.resfile = open("data/"+self.trainsessionname +".dat","a+")
        self.elapsedtime = 0 # elapsed time for this experiment

        self.win_width = int((block_width+block_xdistance) * self.brick_cols + block_xdistance )
        self.win_height = 480

        pygame.init()
        pygame.display.set_caption('Breakout')
        
        #allows for holding of key
        pygame.key.set_repeat(1,0)

        self.screen = pygame.display.set_mode([self.win_width,self.win_height])
        self.myfont = pygame.font.SysFont("Arial",  30)

        self.se_brick = None
        self.se_wall = None
        self.se_paddle = None

        
        
    def init(self, agent):  # init after creation (uses args set from cli)

        print('init ', self.sound_enabled)
        if (self.sound_enabled):
            self.se_brick = pygame.mixer.Sound('brick_hit.wav')
            print('self.se_brick loaded')
            self.se_wall = pygame.mixer.Sound('sound/wall_hit.wav')
            self.se_paddle = pygame.mixer.Sound('sound/paddle_hit.wav')
        if (not self.gui_visible):
            pygame.display.iconify()

        self.agent = agent
        self.setStateActionSpace()
        self.agent.init(self.nstates, self.nactions)
        self.agent.set_action_names(self.action_names)


    def setRandomSeed(self,seed):
        random.seed(seed)
        np.random.seed(seed)

    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime, self.agent.SA_failure]
         
    def loaddata(self,data):
        self.iteration = data[0]
        self.hiscore = data[1]
        self.hireward = data[2]
        self.elapsedtime = data[3]
        try:
            self.agent.SA_failure = data[4]
        except:
            print('WARNING: Cannot load SA_failure data')
    
    def initBricks(self):
        self.bricks = []
        self.bricksgrid = np.zeros((self.brick_cols,self.brick_rows))
        for i in range(0,self.brick_cols):
            for j in range(0,self.brick_rows):
                temp = Brick(i,j)
                self.bricks.append(temp)
                self.bricksgrid[i][j]=1

        
    def reset(self):
        self.ball_x = self.win_width/2
        self.ball_y = self.win_height-100-ball_radius
        self.ball_speed_x = self.init_ball_speed_x
        self.ball_speed_y = self.init_ball_speed_y

        self.randomAngle('i')

        self.paddle_x = self.win_width/2
        self.paddle_y = self.win_height-20
        self.paddle_speed = 10 # same as resolution
        #self.paddle_vec = 0
        self.com_vec = 0

        self.score = 0
        self.ball_hit_count = 0
        self.paddle_hit_count = 0
        self.brick_hit_count = 0
        self.cumreward = 0
        self.gamman = 1.0 # cumulative gamma over time

        self.paddle_hit_without_brick = 0
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        
        self.prev_state = None # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.newstate = True # new state reached
        self.numactions = 0 # number of actions in this run
        self.iteration += 1

        self.agent.optimal = self.optimalPolicyUser or (self.iteration%100)==0 # False #(random.random() < 0.5)  # choose greedy action selection for the entire episode
        
        self.initBricks()

        # firing variables
        self.fire_posx = 0
        self.fire_posy = 0
        self.fire_speedy = 0 # 0 = not firing, <0 firing up




    def goal_reached(self):
        return len(self.bricks) == 0
        
        
    def update(self, a):
        
        self.command = a

        self.prev_state = self.getstate() # remember previous state
        
        #print(" == Update start %d" %self.prev_state)
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        #print('self.current_reward = 0')
        self.numactions += 1
        self.last_brikcsremoved = []

        while (self.prev_state == self.getstate()):
        
            if (self.firstAction):
                self.current_reward += self.STATES['Init']
                self.firstAction = False
            
            if self.command == 0:  # not moving
                # do nothing
                self.current_reward += self.STATES['PaddleNotMoving']
                pass
            elif self.command == 1:  # moving left
                self.paddle_x -= self.paddle_speed
            elif self.command == 2:  # moving right
                self.paddle_x += self.paddle_speed                

            if self.paddle_x < 0:
                self.paddle_x = 0
            if self.paddle_x > self.screen.get_width() - paddle_width:
                self.paddle_x = self.screen.get_width() - paddle_width

            if self.command == 3:  # fire
                if (self.fire_speedy==0):
                    self.fire_posx = self.paddle_x + paddle_width/2
                    self.fire_posy = self.paddle_y
                    if (self.init_ball_speed_y>0):
                        self.fire_speedy = -self.init_ball_speed_y*2
                    else:
                        self.fire_speedy = -10


            self.current_reward += self.STATES['Alive']
            ##MOVE THE BALL
            self.ball_y += self.ball_speed_y
            self.ball_x += self.ball_speed_x

            # firing
            if (self.fire_speedy < 0):
                self.fire_posy += self.fire_speedy

            self.hitDetect()

        #print(" ** Update end - state: %d prev: %d" %(self.getstate(),self.prev_state))

    def randomAngle(self, ev):
        if (ev=='i'): # init
            self.randomAngle1()
        if (ev=='b'): # brick hit
            self.randomAngle2()
        if (ev=='i'): # paddle hit
            self.randomAngle3()
        
    def randomAngle1(self):
        if (not self.deterministic):
            ran = random.uniform(0.75, 1.5)
            self.ball_speed_x *= ran
            #print("random ball_speed_x = %.2f" %self.ball_speed_x)

    def randomAngle2(self):
        if (not self.deterministic):
            ran = random.uniform(0.0, 1.0)
            if (random.uniform(0.0, 1.0) < 0.5):
                self.ball_speed_x *= -1
            #print("random ball_speed_x = %.2f" %self.ball_speed_x)

    def randomAngle3(self):
        if (not self.deterministic):
            ran = random.uniform(0.0, 1.0)
            if (random.uniform(0.0, 1.0) < 0.1):
                self.ball_speed_x *= 0.75
            elif (random.uniform(0.0, 1.0) > 0.9):
                self.ball_speed_x *= 1.5
            sign = self.ball_speed_x/abs(self.ball_speed_x)    
            self.ball_speed_x = min(self.ball_speed_x,6)*sign
            self.ball_speed_x = max(self.ball_speed_x,0.5)*sign
            #print("random ball_speed_x = %.2f" %self.ball_speed_x)

            
    def hitDetect(self):
        ##COLLISION DETECTION
        ball_rect = pygame.Rect(self.ball_x-ball_radius, self.ball_y-ball_radius, ball_radius*2,ball_radius*2) #circles are measured from the center, so have to subtract 1 radius from the x and y
        paddle_rect = pygame.Rect(self.paddle_x, self.paddle_y, paddle_width, paddle_height)

        fire_rect = pygame.Rect(self.fire_posx-1, self.fire_posy-1, 3, 3)

        # print("fire pos %d %d - spd %d" %(self.fire_posx, self.fire_posy, self.fire_speedy))

        # TERMINATION OF EPISODE
        if (not self.finished):
            #check if the ball is off the bottom of the self.screen
            end1 = self.ball_y > self.screen.get_height() - ball_radius
            end2 = self.goal_reached()
            end3 = self.paddle_hit_without_brick == 30
            end3b = self.numactions > 500 * self.brick_cols
            end4 = len(self.bricks) == 0
            if (end1 or end2 or end3 or end3b or end4):
                if (pygame.display.get_active() and (not self.se_wall is None)):
                    self.se_wall.play()
                if (end1):    
                    self.current_reward += self.STATES['Dead']
                if (end2):
                    self.ngoalreached += 1
                    self.current_reward += self.STATES['Goal']

                self.finished = True # game will be reset at the beginning of next iteration
                return 
        

        #for screen border
        if self.ball_y < ball_radius:
            self.ball_y = ball_radius
            self.ball_speed_y = -self.ball_speed_y
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_wall.play()
        if self.ball_x < ball_radius:
            self.ball_x = ball_radius
            self.ball_speed_x = -self.ball_speed_x
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_wall.play()
        if self.ball_x > self.screen.get_width() - ball_radius:
            self.ball_x = self.screen.get_width() - ball_radius
            self.ball_speed_x = -self.ball_speed_x
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_wall.play()

        #for paddle
        if ball_rect.colliderect(paddle_rect):
            if (self.paddle_complex_bump):
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width/2))
                if (dbp<20):
                    #print 'straight'
                    if (self.ball_speed_x<-5):
                        self.ball_speed_x += 2
                    elif (self.ball_speed_x>5):
                        self.ball_speed_x -= 2
                    elif (self.ball_speed_x<=-0.5): 
                        self.ball_speed_x += 0.5
                    elif (self.ball_speed_x>=0.5): 
                        self.ball_speed_x -= 0.5

                dbp = math.fabs(self.ball_x-(self.paddle_x+0))
                if (dbp<10):
                    #print 'left' 
                    self.ball_speed_x = -abs(self.ball_speed_x)-1
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width))
                if (dbp<10):
                    #print 'right'
                    self.ball_speed_x = abs(self.ball_speed_x)+1

            elif (self.paddle_normal_bump):
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width/2))
                if (dbp<20):
                    #print 'straight'
                    if (self.ball_speed_x!=0):
                        self.ball_speed_x = 2*abs(self.ball_speed_x)/self.ball_speed_x
                dbp = math.fabs(self.ball_x-(self.paddle_x+0))
                if (dbp<20):
                    #print 'left' 
                    self.ball_speed_x = -5
                    self.randomAngle('p')
                dbp = math.fabs(self.ball_x-(self.paddle_x+paddle_width))
                if (dbp<20):
                    #print 'right'
                    self.ball_speed_x = 5
                    self.randomAngle('p')
                    
            self.ball_speed_y = - abs(self.ball_speed_y)
            self.current_reward += self.STATES['Hit']
            self.ball_hit_count +=1
            self.paddle_hit_count +=1
            self.paddle_hit_without_brick += 1
            if (pygame.display.get_active() and (not self.se_wall is None)):
                self.se_paddle.play()

            # reset after paddle hits the ball
            if len(self.bricks) == 0:
                # print("  -- %6d  *** New level ***" %(self.iteration) )
                self.initBricks()
                self.ball_speed_y = self.init_ball_speed_y
                
        #for bricks
        #hitbrick = False
        for brick in self.bricks:
            if brick.rect.colliderect(ball_rect):
                #print 'brick hit ',brick.i,brick.j
                if ((not self.se_brick is None)):  #pygame.display.get_active() and 
                    self.se_brick.play()
                    #print('self.se_brick.play()')

                self.score = self.score + 1
                self.brick_hit_count += 1
                self.bricks.remove(brick)
                self.last_brikcsremoved.append(brick)
                self.bricksgrid[(brick.i,brick.j)] = 0
                self.ball_speed_y = -self.ball_speed_y
                self.current_reward += self.STATES['Scores']
                self.paddle_hit_without_brick = 0
                #print("bricks left: %d" %len(self.bricks))
                break

        #firing
        if (self.fire_posy < 5):
            #reset
            self.fire_posx = 0
            self.fire_posy = 0
            self.fire_speedy = 0

        for brick in self.bricks:
            if brick.rect.colliderect(fire_rect):
                #print 'brick hit with fire ',brick.i,brick.j
                if (pygame.display.get_active() and (not self.se_wall is None)):
                    self.se_brick.play()
                self.score = self.score + 1
                self.bricks.remove(brick)
                self.last_brikcsremoved.append(brick)
                self.bricksgrid[(brick.i,brick.j)] = 0
                self.current_reward += self.STATES['Scores']
                self.paddle_hit_without_brick = 0
                #print("bricks left: %d" %len(self.bricks))
                # reset firing
                self.fire_posx = 0
                self.fire_posy = 0
                self.fire_speedy = 0
                break

        if self.brick_hit_count > 0:
            self.randomAngle('b')
            self.brick_hit_count = 0


    def input(self):
        self.isPressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('pygame quit event')
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.command = 1
                    self.isPressed = True
                elif event.key == pygame.K_RIGHT:
                    self.command = 2
                    self.isPressed = True
                elif event.key == pygame.K_x: # Fire
                    self.command = 3
                    self.isPressed = True
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    print("Game paused: %d" %self.pause)
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
                    self.agent.debug = False
                elif event.key == pygame.K_g:
                    self.sleeptime = 0.0
                    self.agent.debug = False
                elif event.key == pygame.K_o:
                    self.optimalPolicyUser = not self.optimalPolicyUser
                    print("Best policy: %d" %self.optimalPolicyUser)
                elif event.key == pygame.K_q:
                    self.userquit = True
                    print("User quit !!!")
                    
        if not self.isPressed:
            self.command = 0

        return True

    def getUserAction(self):
        return self.command

    def getreward(self):
        r = self.current_reward
        failed = self.RA is not None and self.RA.current_node==self.RA.RAFail  # FAIL RA state
        if (self.current_reward>0 and failed):  
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
            
        s = 'Iter %6d, sc: %3d, p_hit: %3d, na: %4d, r: %5d  %c' %(self.iteration, self.score, self.paddle_hit_count,self.numactions, self.cumreward, ch)

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
        pgoal = 0
        if (self.iteration%numiter==0):
            #self.doSave()
            pgoal = float(self.ngoalreached*100)/numiter
            print('-----------------------------------------------------------------------')
            print("%s %6d/%4d avg last 100: reward %.1f | score %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100/100), float(self.cumscore100)/100, pgoal))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0
            self.cumscore100 = 0
            self.ngoalreached = 0
            

        sys.stdout.flush()
        
        self.vscores.append(self.score)
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
            cmd = 'o'
        #s = '%d %s' %(x,cmd)
        s = '%s' %(cmd)
        count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
        self.screen.blit(count_label, (60, 10))
        #count_label = self.myfont.render(str(self.ball_speed_y), 100, pygame.color.THECOLORS['brown'])
        #self.screen.blit(count_label, (160, 10))

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-200, 10))
        if (self.agent.optimal):
            opt_label = self.myfont.render("Best", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(opt_label, (self.win_width-100, 10))
            
        for brick in self.bricks:
            pygame.draw.rect(self.screen,grey,brick.rect,0)
        pygame.draw.circle(self.screen, orange, [int(self.ball_x), int(self.ball_y)], ball_radius, 0)
        pygame.draw.rect(self.screen, grey, [self.paddle_x, self.paddle_y, paddle_width, paddle_height], 0)

        # print("fire %d %d %d" %(self.fire_posx, self.fire_posy,self.fire_speedy))
        if (self.fire_speedy<0):
            pygame.draw.rect(self.screen, red, [self.fire_posx, self.fire_posy, 5, 5], 0)

        pygame.display.update()


    def quit(self):
        self.resfile.close()
        pygame.quit()

    # To be implemented by sub-classes    
    def setStateActionSpace(self):
        print('ERROR: this function must be overwritten by subclasses')
        sys.exit(1)
        
    def getstate(self):
        print('ERROR: this function must be overwritten by subclasses')
        sys.exit(1)


#
# Breakout with standard definition of states
#
class BreakoutN(Breakout):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test'):
        Breakout.__init__(self,brick_rows, brick_cols, trainsessionname)
        
    def setStateActionSpace(self):
        self.n_ball_x = int(self.win_width/resolutionx)+1
        self.n_ball_y = int(self.win_height/resolutiony)+1
        self.n_ball_dir = 10 # ball going up (0-5) or down (6-9)
                        # ball going left (1,2) straight (0) right (3,4)
        self.n_paddle_x = int(self.win_width/resolutionx)+1

        self.nactions = 3  # 0: not moving, 1: left, 2: right
        if (self.fire_enabled):
            self.nactions = 4  # 3: fire
        
        self.nstates = self.n_ball_x * self.n_ball_y * self.n_ball_dir * self.n_paddle_x
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
        
        x = ball_x  + self.n_ball_x * ball_y + (self.n_ball_x*self.n_ball_y) * ball_dir + (self.n_ball_x*self.n_ball_y*self.n_ball_dir) * paddle_x
        
        return int(x)


#
# Breakout with simplified definition of states
#

class BreakoutS(Breakout):

    def __init__(self, brick_rows=3, brick_cols=3, trainsessionname='test'):
        Breakout.__init__(self,brick_rows, brick_cols, trainsessionname)

    def setStateActionSpace(self):
        self.n_diff_paddle_ball = int(2*self.win_width/resolutionx)+1

        self.nactions = 3  # 0: not moving, 1: left, 2: right
        if (self.fire_enabled):
            self.nactions = 4  # 3: fire
        
        self.nstates = self.n_diff_paddle_ball
        print('Number of states: %d' %self.nstates)
        print('Number of actions: %d' %self.nactions)

        
    def getstate(self):
        resx = resolutionx 

        diff_paddle_ball = int((self.ball_x-self.paddle_x+self.win_width)/resx)
        
        return diff_paddle_ball


