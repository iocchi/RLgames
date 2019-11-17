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
dgrey = [120,120,120]
orange = [180,100,20]
red = [200,0,0]
pink = [250, 150, 150]
green = [0,200,0]
lgreen = [60,250,60]
dgreen = [0,100,0]
blue = [0,0,250]
lblue = [80,200,200]
brown = [140, 100, 40]
dbrown = [100, 80, 0]
gold = [230, 215, 80]
yellow = [210, 250, 80]


class TaskExecutor(object):

    def __init__(self, rows=5, cols=5, trainsessionname='test'):
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
        
        self.differential = False
        self.initial_pos_x = 0
        self.initial_pos_y = 0
        self.initial_pos_th = 90

        self.sleeptime = 0.0
        self.command = 0
        self.iteration = 0
        self.score = 0
        self.numactions = 0
        self.cumreward = 0
        self.cumreward100 = 0 # cum reward for statistics
        self.cumscore100 = 0 
        self.ngoalreached = 0
        
        self.nactionlimit = 1000
        self.ntaskactionslimit = 1000
        self.turnslimit = 10 # max consecutive turns allowed
        self.useslimit = 100 # max consecutive uses allowed
        
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

        self.RA_exploration_enabled = False # enable automatic options
        self.RA_visits = {} # number of visits for each RA state
        self.RA_success = {} # number of good transitions for each RA state

        if (self.cols>10):
            self.win_width += self.size_square * (self.cols-10)
        if (self.rows>10):
            self.win_height += self.size_square * (self.rows-10)

        pygame.init()

        #allows for holding of key
        #pygame.key.set_repeat(1,0)

        # self.reset()  called by game engine

        self.screen = pygame.display.set_mode([self.win_width,self.win_height])
        self.myfont = pygame.font.SysFont("Arial",  30)

        # to be set by sub-classes
        #self.locations = LOCATIONS
        #self.action_names = ACTION_NAMES
        #self.tasks = TASKS
        #self.reward_states = REWARD_STATES
        self.maxitemsheld = 1 # max number of items agent can hold

        


    def ntaskstates(self):
        r = 1
        for t in self.tasks.keys():
            tl = self.tasks[t]
            for l in tl: 
                r *= len(l)+1
        return r

        
    def init(self, agent):  # init after creation (uses args set from cli)
        if (not self.gui_visible):
            pygame.display.iconify()

        # number of states
        ns = self.rows * self.cols
        if self.differential:
            ns *= 4
            self.nactionlimit *= 5
            #self.ntaskactionslimit *= 4

        # number of actions        
        self.nactions = len(self.action_names)
    
        self.agent = agent
        ns *= self.ntaskstates()
        print('Number of states: %d' %ns)
        print('Number of actions: %d' %self.nactions)
        self.agent.init(ns, self.nactions) # 1 for RA not used here
        self.agent.set_action_names(self.action_names)

    def setRandomSeed(self,seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):       
        self.pos_x = self.initial_pos_x
        self.pos_y = self.initial_pos_y
        self.pos_th = self.initial_pos_th
        self.consecutive_turns = 0
        self.consecutive_uses = 0
        
        self.reset_tasks()
        self.has = []
        
        self.score = 0
        self.gamman = 1.0 # cumulative gamma over time
        self.cumreward = 0
        self.cumscore = 0  
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state

        self.prev_state = None # previous state
        self.firstAction = True # first action of the episode
        self.finished = False # episode finished
        self.newstate = True # new state reached
        self.numactions = 0 # number of actions in this episode
        self.ntaskactions = 0 # number of actions for this task-part of episode 
        self.iteration += 1

        self.agent.optimal = self.optimalPolicyUser or (self.iteration%100)==0 # False #(random.random() < 0.5)  # choose greedy action selection for the entire episode
        
        self.current_RA_state = 0
        self.last_RA_state = -1
        self.state_changed = False
        
        # RA exploration
        self.RA_exploration()

        
    def reset_tasks(self):
        # RA state of each sub-task
        self.task_state = {}
        for t in self.tasks.keys():
            tl = self.tasks[t]
            i = 0
            for l in tl:
                self.task_state[(t,i)]=0
                i += 1
        self.actionlocation = []
        self.ntaskactions = 0
        self.taskscompleted = []

    def reset_partial_tasks(self):
        # reset state of each sub-task
        for t in self.tasks.keys():
            ltl = self.tasks[t]
            i = 0
            for tl in ltl:
                if t in self.taskscompleted:
                    self.task_state[(t,i)]=len(tl)
                elif self.task_state[(t,i)] < len(tl):
                    #print('reset task %s' %t)
                    self.task_state[(t,i)]=0
                i += 1
        self.actionlocation = []


    def encode_task_state(self):
        r = 0
        b = 1
        for t in self.tasks.keys():
            tl = self.tasks[t]
            i = 0
            for l in tl:
                r += b * self.task_state[(t,i)]
                b *= len(l)+1
                i += 1
#            print('    ---  encode task state  ',t , self.task_state[t])
#        print('    ---  encode task state final: ', r)
        return r
        
    def getstate(self):
        x = self.pos_x + self.cols * self.pos_y 
#        print '-----'
#        print (self.pos_x,self.pos_y,self.pos_th/90,self.encode_task_state())
#        print ' +++ state: ',x
        n = (self.rows * self.cols)
        if (self.differential):
            x += (self.pos_th/90) * n
            n *= 4
        x += n * self.encode_task_state()
#        print ' +++ state: ',x
#        print ' === state: ',x,'\n'
        return x        

    def goal_reached(self):
        r = self.score==len(self.tasks.keys())
#        print ' --- goal reached - score ', self.score
#        if r:
#            print ' --- goal reached!!!'
        return r
        
    def savedata(self):
        return [self.iteration, self.hiscore, self.hireward, self.elapsedtime,
            self.RA_visits, self.RA_success]

    def loaddata(self,data):
         self.iteration = data[0]
         self.hiscore = data[1]
         self.hireward = data[2]
         self.elapsedtime = data[3]
         self.RA_visits = data[4]
         self.RA_success = data[5]

    def itemat(self, x, y): # which item is in this location
        r = None
        for t in self.locations:
            if (t[2]==x and t[3]==y):
                r = t[0]
                break
        return r

    # check if this action progresses any task
    # check if any task finished and resets other sub-tasks
    def check_action_task(self,a,what=None):  
        r = 0 # reward to return
        self.state_changed = False # if RA state is changed
        if (what!=None):
            act = a+"_"+what
        else:
            act = a
        if not self.isAuto:
            print("checking action %s" %act)
        for t in self.tasks.keys():
            ltl = self.tasks[t]
            i = 0
            for tl in ltl: # action list for this task
                ts = self.task_state[(t,i)]
                if not self.isAuto:
                    print('  -- task list: %r status: %d' %(tl,ts))
                if (ts<len(tl) and act==tl[ts]):
                    if not self.isAuto:
                        print('*** good action for %s ***' %t)
                    self.actionlocation.append((self.pos_x, self.pos_y))
                    self.task_state[(t,i)] += 1 # go to next task state
                    r += self.reward_states['TaskProgress']
                    if (self.task_state[(t,i)] == len(tl)):
                        if not self.isAuto:
                            print("!!!Task %s completed!!!" %t)
                        self.taskscompleted.append(t)
                        r += self.reward_states['TaskComplete']
                        self.state_changed = True
                        #print("state changed")
                        self.score += 1
                        self.reset_partial_tasks()
                i += 1
        #print('   ... reward %d' %r)
        return r 

    def current_successrate(self):
        s = 0.0
        v = 1.0
        if (self.current_RA_state in self.RA_success):
            s = float(self.RA_success[self.current_RA_state])
        if (self.current_RA_state in self.RA_visits):
            v = float(self.RA_visits[self.current_RA_state])
        #print "   -- state %d - success rate: %f / %f" %(self.current_RA_state,s,v)
        return s/v
        
    def RA_exploration(self):
        if (not self.RA_exploration_enabled):
            return
        # update success/visit
        #print("update success/visit")
        self.current_RA_state = self.encode_task_state()
        if (self.current_RA_state in self.RA_visits):
            self.RA_visits[self.current_RA_state] += 1
        else:
            self.RA_visits[self.current_RA_state] = 1

        if (self.last_RA_state>=0 and self.last_RA_state in self.RA_success):
            self.RA_success[self.last_RA_state] += 1
        else:
            self.RA_success[self.last_RA_state] = 1
        self.last_RA_state = self.current_RA_state
    
        #print "RA state: ",self.current_RA_state
        success_rate = max(min(self.current_successrate(),0.9),0.1)
        #print "RA exploration policy: current state success rate ",success_rate
        er = random.random()
        self.agent.option_enabled = (er<success_rate)
        #print "RA exploration policy: optimal ",self.agent.partialoptimal, "\n"

        
    def update(self, a):
        
        self.command = a
        
        self.prev_state = self.getstate() # remember previous state
        
        #print " == Update start ",self.prev_state," action",self.command 
        
        self.current_reward = 0 # accumulate reward over all events happened during this action until next different state
        #print('self.current_reward = 0')
        self.numactions += 1 # total number of actions axecuted in this episode
        self.ntaskactions += 1
        # while (self.prev_state == self.getstate()):
        
        if (self.firstAction):
            self.firstAction = False
            self.current_reward += self.reward_states['Init']
        
        newposx = self.pos_x
        newposy = self.pos_y

        if (not self.differential):
            if self.command == 0:  # moving left
                newposx = self.pos_x - 1
            elif self.command == 1:  # moving right
                newposx = self.pos_x + 1
            elif self.command == 2:  # moving up
                newposy = self.pos_y + 1
            elif self.command == 3:  # moving down
                newposy = self.pos_y - 1

        else:
            # differential motion
            if self.command == 0: # turn left
                self.pos_th += 90
                if (self.pos_th >= 360):
                    self.pos_th -= 360
                #print ("left") 
                self.consecutive_turns += 1
                self.current_reward += self.reward_states['Turn']
            elif self.command == 1:  # turn right
                self.pos_th -= 90
                if (self.pos_th < 0):
                    self.pos_th += 360 
                #print ("right")
                self.consecutive_turns += 1
                self.current_reward += self.reward_states['Turn']
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
                else:
                    #print ("forward")
                    self.current_reward += self.reward_states['Forward']
                self.consecutive_turns = 0
                self.consecutive_uses = 0
                newposx = self.pos_x + dx
                newposy = self.pos_y + dy

        if (newposx < 0):
            newposx = 0 
            self.current_reward += self.reward_states['Hit']
        if (newposx >= self.cols):
            newposx = self.cols-1
            self.current_reward += self.reward_states['Hit']
        if (newposy >= self.rows):
            newposy = self.rows-1
            self.current_reward += self.reward_states['Hit']
        if (newposy< 0):
            newposy = 0 
            self.current_reward += self.reward_states['Hit']

        if self.itemat(newposx,newposy)=='obstacle':
            self.current_reward += self.reward_states['Hit']
        else:
            self.pos_x = newposx
            self.pos_y = newposy


        if self.command>=4:
            r = 0
            if (self.command in self.map_actionfns):
                # exec action function
                r = self.map_actionfns[self.command]()
            else:
                print('ERROR: action command %d unknown!!!' %self.command)
            self.current_reward += r
            self.consecutive_uses += 1

        self.current_reward += self.reward_states['Alive']

        # RA exploration   
        if (self.state_changed):  # when task completed
            self.RA_exploration()
            self.state_changed = False

        # check if episode terminated
        if self.goal_reached():
            self.current_reward += self.reward_states['Score']
            self.ngoalreached += 1
            self.finished = True
            
        if (self.numactions>self.nactionlimit):
            self.current_reward += self.reward_states['Dead']
            self.finished = True

        # too many consecutive actions
        if (self.consecutive_turns>self.turnslimit or self.consecutive_uses>self.useslimit or self.ntaskactions > self.ntaskactionslimit):
            self.finished = True
            #if (self.agent.optimal):
            #    #self.finished = True
            #    pass
            #elif (self.agent.partialoptimal):
            #    self.agent.partialoptimal = False
                
        #print " ** Update end ",self.getstate(), " prev ",self.prev_state
        


    def input(self):
    
        self.usercommand = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.usercommand = 0
                elif event.key == pygame.K_RIGHT:
                    self.usercommand = 1
                elif event.key == pygame.K_UP:
                    self.usercommand = 2
                elif event.key == pygame.K_DOWN:
                    self.usercommand = 3
                elif event.key == pygame.K_4: # user action
                    self.usercommand = 4
                elif event.key == pygame.K_5: # user action
                    self.usercommand = 5
                elif event.key == pygame.K_6: # user action
                    self.usercommand = 6
                elif event.key == pygame.K_7: # user action
                    self.usercommand = 7
                elif event.key == pygame.K_8: # user action
                    self.usercommand = 8
                elif event.key == pygame.K_9: # user action
                    self.usercommand = 9
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    print("Game paused: %s" %self.pause)
                elif event.key == pygame.K_a:
                    self.isAuto = not self.isAuto
                elif event.key == pygame.K_s:
                    self.sleeptime = 1.0
                elif event.key == pygame.K_d:
                    self.sleeptime = 0.05
                elif event.key == pygame.K_f:
                    self.sleeptime = 0.0
                    self.agent.debug = False
                elif event.key == pygame.K_o:
                    self.optimalPolicyUser = not self.optimalPolicyUser
                    print("Best policy: %s" %self.optimalPolicyUser)
                elif event.key == pygame.K_q:
                    self.userquit = True
                    print("User quit !!!")

        if self.usercommand>=self.nactions:
            self.usercommand = -1

        return not self.userquit

    def getUserAction(self):
        while (self.usercommand<0 and not self.isAuto and not self.userquit):
            self.input()
            time.sleep(0.2)
        if (not self.isAuto):
            self.command = self.usercommand
        return self.command

    def getreward(self):
        r = self.current_reward        
        self.cumreward += self.gamman * r
        self.gamman *= self.agent.gamma
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
            print("%s %6d/%4d avg last 100: reward %d | score %.2f | p goals %.1f %%" %(self.trainsessionname, self.iteration, self.elapsedtime, float(self.cumreward100)/100, float(self.cumscore100)/100, pgoal))
            print('-----------------------------------------------------------------------')
            self.cumreward100 = 0  
            self.cumscore100 = 0 
            self.ngoalreached = 0

        sys.stdout.flush()

        self.resfile.write("%d,%d,%d,%d,%d,%d,%d\n" % (self.iteration, self.elapsedtime, self.score, self.cumreward, self.goal_reached(),self.numactions,self.agent.optimal))
        self.resfile.flush()


    def draw(self):
        self.screen.fill(white)

        score_label = self.myfont.render(str(self.score), 100, pygame.color.THECOLORS['black'])
        self.screen.blit(score_label, (20, 10))

        #count_label = self.myfont.render(str(self.paddle_hit_count), 100, pygame.color.THECOLORS['brown'])
        #self.screen.blit(count_label, (70, 10))

        if self.command<self.nactions:
            x = self.getstate()
            cmd = self.action_names[self.command]
            s = '%d %s' %(x,cmd)
            count_label = self.myfont.render(s, 100, pygame.color.THECOLORS['brown'])
            self.screen.blit(count_label, (60, 10))
        
        sinv = ''
        for t in self.tasks.keys():
            ltl = self.tasks[t]
            i = 0
            st = '-'
            for tl in ltl:
                if (self.task_state[(t,i)] == len(tl)):
                    st = '*'
            sinv += st
            
        inv_label = self.myfont.render(sinv, 100, pygame.color.THECOLORS['blue'])
        self.screen.blit(inv_label, (200, 10))

        if self.isAuto is True:
            auto_label = self.myfont.render("Auto", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(auto_label, (self.win_width-160, 10))
        if (self.agent.optimal):
            opt_label = self.myfont.render("Best", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(opt_label, (self.win_width-80, 10))
        elif (self.agent.option_enabled):
            opt_label = self.myfont.render("PB", 100, pygame.color.THECOLORS['red'])
            self.screen.blit(opt_label, (self.win_width-80, 10))

        # grid
        for i in range (0,self.cols+1):
            ox = self.offx + i*self.size_square
            pygame.draw.line(self.screen, black, [ox, self.offy], [ox, self.offy+self.rows*self.size_square])
        for i in range (0,self.rows+1):
            oy = self.offy + i*self.size_square
            pygame.draw.line(self.screen, black, [self.offx , oy], [self.offx + self.cols*self.size_square, oy])

        # world elements
        for t in self.locations:
            col = t[1]
            u = t[2]
            v = t[3]
            dx = int(self.offx + u * self.size_square)
            dy = int(self.offy + (self.rows-v-1) * self.size_square)
            sqsz = (dx+5,dy+5,self.size_square-10,self.size_square-10)
            pygame.draw.rect(self.screen, col, sqsz)
            if ((u,v) in self.actionlocation):
                pygame.draw.rect(self.screen, pygame.color.THECOLORS['black'], (dx+15,dy+15,self.size_square-30,self.size_square-30))

        # agent position
        dx = int(self.offx + self.pos_x * self.size_square)
        dy = int(self.offy + (self.rows-self.pos_y-1) * self.size_square)

        pygame.draw.circle(self.screen, orange, [dx+self.size_square//2, dy+self.size_square//2], 2*self.radius, 0)


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

        pygame.draw.circle(self.screen, pygame.color.THECOLORS['black'], [int(dx+self.size_square/2+ox), int(dy+self.size_square/2+oy)], 5, 0)
        
         
        pygame.display.update()



    def quit(self):
        self.resfile.close()
        pygame.quit()

