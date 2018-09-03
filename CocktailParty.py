import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs
import TaskExecutor
from TaskExecutor import *

ACTION_NAMES = ['<-','->','^','v','g','d'] 
# 0: left, 1: right, 2: up, 3: down, 4: get, 5: deliver

#RESOURCES = ['coke', 'beer', 'chips', 'biscuits' ]  # for get actions

LOCATIONS = [ ('coke',red,1,1), ('beer',gold,2,3), 
    ('chips',yellow,3,1), ('biscuits',brown,0,3), 
    ('john',blue,4,2), ('mary',pink,1,4) 
]


TASKS = { 
    'serve_drink_john': [ ['get_coke', 'deliver_john'], ['get_beer', 'deliver_john'] ],
    'serve_drink_mary': [ ['get_coke', 'deliver_mary'], ['get_beer', 'deliver_mary'] ],
    'serve_snack_john': [ ['get_chips', 'deliver_john'], ['get_biscuits', 'deliver_john'] ],
    'serve_snack_mary': [ ['get_chips', 'deliver_mary'], ['get_biscuits', 'deliver_mary'] ]
}

TASK1 = { 
    'serve': [ ['get_coke', 'deliver_john'], ['get_beer', 'deliver_john'],
        ['get_coke', 'deliver_mary'], ['get_beer', 'deliver_mary'],
        ['get_chips', 'deliver_john'], ['get_biscuits', 'deliver_john'],
        ['get_chips', 'deliver_mary'], ['get_biscuits', 'deliver_mary'] ]
}

REWARD_STATES = {
    'Init':0,
    'Alive':0,
    'Dead':0,
    'Score':1000,
    'Hit':0,
    'Forward':0,
    'Turn':0,
    'BadGet':0,        
    'BadDeliver':0, 
    'TaskProgress':100,
    'TaskComplete':1000
}



class CocktailParty(TaskExecutor):

    def __init__(self, rows=5, cols=5, trainsessionname='test'):
        global ACTION_NAMES, LOCATIONS, TASKS, REWARD_STATES
        TaskExecutor.__init__(self, rows, cols, trainsessionname)
        self.locations = LOCATIONS
        self.action_names = ACTION_NAMES
        self.tasks = TASKS
        self.reward_states = REWARD_STATES
        self.maxitemsheld = 1
        self.map_actionfns = { 4: self.doget, 5: self.dodeliver }
        self.onetask = False
        if (cols>5):
            self.move('john',cols-1,2)
            self.move('mary',1,rows-1)
            self.move('coke',cols/2, rows/2)
            self.move('chips',3,rows/2)

    def move(self, what, xnew, ynew):
        f = None
        n = None
        for t in self.locations:
            if (t[0]==what):
                f = t
                n = (t[0],t[1],xnew,ynew)
        self.locations.remove(f)
        self.locations.append(n)


    def setOneTask(self):
        self.onetask = True
        self.tasks = TASK1

    def doget(self):
        what = self.itemat(self.pos_x, self.pos_y)
        if what!=None and not self.isAuto:
            print "get: ",what
        if (what==None):
            r = self.reward_states['BadGet']
        elif (len(self.has)==self.maxitemsheld):
            r = self.reward_states['BadGet']
        else:
            self.has.append(what)
            r = self.check_action_task('get',what)
        return r

    def dodeliver(self):
        what = self.itemat(self.pos_x, self.pos_y)
        if what!=None and not self.isAuto:
            print "deliver %r to %s " %(self.has,what)
        if (what==None):
            r = self.reward_states['BadDeliver']
        elif (len(self.has)==0):
            r = self.reward_states['BadDeliver']
        else:
            self.has = []
            r = self.check_action_task('deliver',what)
        return r


