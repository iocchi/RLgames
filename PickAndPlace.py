import pygame, sys
import numpy as np
import atexit
import random
import time
import math
from math import fabs
import TaskExecutor
from TaskExecutor import *

ACTION_NAMES = ['<-','->','^','v','k','c','h'] 
# 0: left, 1: right, 2: up, 3: down, 4: pick, 5: place, 6: home

LOCATIONS = [ ('home',yellow,0,0) ] # ('item',green,1,0) , ('shelf',red,9,0) added dynamically


TASKS = { 
    'pick_and_place': [ ['pick_item', 'place_shelf', 'home'] ]
}

REWARD_STATES = {
    'Init':0,
    'Alive':0,
    'Dead':0,
    'Score':1000,
    'Hit':0,
    'Forward':0,
    'Turn':0,
    'BadPick':0,        
    'BadPlace':0, 
    'BadHome':0, 
    'TaskProgress':100,
    'TaskComplete':1000
}



class PickAndPlace(TaskExecutor):

    def __init__(self, rows=10, cols=9, trainsessionname='test'):
        global ACTION_NAMES, LOCATIONS, TASKS, REWARD_STATES
        TaskExecutor.__init__(self, rows, cols, trainsessionname)
        LOCATIONS.append(('item',green,cols/2,0))
        LOCATIONS.append(('shelf',red,cols-1,0))
        self.locations = LOCATIONS
        self.action_names = ACTION_NAMES
        self.tasks = TASKS
        self.reward_states = REWARD_STATES
        self.maxitemsheld = 1
        self.map_actionfns = { 4: self.dopick, 5: self.doplace, 6: self.dohome }


    def dopick(self):
        what = self.itemat(self.pos_x, self.pos_y)
        if what!=None and not self.isAuto:
            print("pick: ",what)
        if (what!='item'):
            r = self.reward_states['BadPick']
        elif (len(self.has)==self.maxitemsheld):
            r = self.reward_states['BadPick']
        else:
            self.has.append(what)
            r = self.check_action_task('pick',what)
        return r

    def doplace(self):
        what = self.itemat(self.pos_x, self.pos_y)
        if what!=None and not self.isAuto:
            print("place %r to %s " %(self.has,what))
        if (what==None):
            r = self.reward_states['BadPlace']
        elif (len(self.has)==0):
            r = self.reward_states['BadPlace']
        else:
            self.has = []
            r = self.check_action_task('place',what)
        return r

    def dohome(self):
        what = self.itemat(self.pos_x, self.pos_y)
        if what!=None and not self.isAuto:
            print("home to %s " %what)
        if (what!='home'):
            r = self.reward_states['BadHome']
        else:
            r = self.check_action_task('home')
        return r
            

